import os
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
srcdir = parentdir + '/src'
os.sys.path.insert(1, srcdir)

import numpy as np
import copy
import time
import pybullet as p
import pybullet_data
import pinocchio as se3
import matplotlib
import tsid
import numpy.matlib as matlib
from numpy import nan
from numpy.linalg import norm as norm
import commands

import matplotlib.pyplot as plt
from pinbullet_wrapper import PinBulletWrapper
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.utils import zero
from romeo_foot import RomeoFoot

from dotmap import DotMap
import pickle

font = {'family' : 'serif',
        'size'   : 22}

matplotlib.rc('font', **font)
np.set_printoptions(precision=3, linewidth=200, suppress=True)

BMODE = p.GUI
LOGGING = True
LOCAL = True

STEP_DURATION = 800
N_STEPS = 6
N_SIMULATION = N_STEPS * STEP_DURATION
DOUBLE_SUPPORT_DURATION = 5

DELAY = 4
NS3 = True
SCENARIO = 'building'
IDX = 5

DELAY_START = 0
ONBOARD_WINDOW = 50

PRELANDING = 10
PRELIFTING = 10

ONBOARD_FREQ = 5

# compute delayed indices
DELAYED_IDX = []

if NS3:
    ns3_path = parentdir + '/data/ns3/' + SCENARIO + '/delayed_idx/'
    DELAYED_IDX = np.load(ns3_path + str(IDX) +'.npy')

else:
    for i in range(N_SIMULATION):
        if i >= DELAY_START:
            delayed_idx = np.maximum(i-DELAY, DELAY_START)
        else:
            delayed_idx = i
        DELAYED_IDX.append(delayed_idx)

DELAYED_IDX_RAW = copy.copy(DELAYED_IDX)

Com = np.load(parentdir+'/data/com.npy')
foot_steps = np.load(parentdir+'/data/foot_steps.npy')
left_foot_steps = foot_steps[1::2]
right_foot_steps = foot_steps[0::2]

mu = 0.5                            # friction coefficient
rf_frame_name = "RAnkleRoll"        # right foot frame name
lf_frame_name = "LAnkleRoll"        # left foot frame name

w_com = 1.0                     # weight of center of mass task
w_posture = 1e-3                # weight of joint posture task
w_forceRef = 1e-5               # weight of force regularization task
w_foot_motion = 1.0

kp_contact = 10.0               # proportional gain of contact constraint
kd_contact = 7.0
kp_com = 10.0                   # proportional gain of center of mass task
kp_posture = 50.0               # proportional gain of joint posture task
kp_foot_motion = 30.0
kd_foot_motion = 1000.0

sigma_q = 0
sigma_v = 0 # measurement noise

path = parentdir + '/models/romeo'
robot_urdf = path + '/urdf/romeo.urdf'
plane_urdf = parentdir + '/models/plane.urdf'
vector = se3.StdVec_StdString()
vector.extend(item for item in path)
robot = tsid.RobotWrapper(robot_urdf, vector, se3.JointModelFreeFlyer(), False)
pinocchio_robot = se3.buildModelFromUrdf(robot_urdf, se3.JointModelFreeFlyer())

physicsClient = p.connect(BMODE)#or p.DIRECT for non-graphical version
dt = 1./1000.
p.setTimeStep(dt)
p.setGravity(0,0,-9.81)
robot_id = p.loadURDF(robot_urdf,[0,0,0])
plane_id = p.loadURDF(plane_urdf,[0,0,-0.88], useFixedBase=1)

joint_names = [n for n in robot.model().names]
del joint_names[0]
del joint_names[0]
endeff_names = ["l_sole", "r_sole"]
pbwrapper = PinBulletWrapper(robot_id, pinocchio_robot, joint_names, endeff_names)

p.changeDynamics(plane_id, -1, restitution=0.99,
                 lateralFriction=1.0, spinningFriction=1.0, rollingFriction=1.0,
                 contactStiffness=1e6, contactDamping=2e3)
p.resetJointState(robot_id, 20, np.pi/2)
p.resetJointState(robot_id, 27, np.pi/2)

q, v = pbwrapper.get_state()
assert robot.model().existFrame(rf_frame_name)
assert robot.model().existFrame(lf_frame_name)

t = 0.

invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
invdyn.computeProblemData(t, q, v)
data = invdyn.data()

comTask = tsid.TaskComEquality("task-com", robot)
comTask.setKp(kp_com * np.matrix(np.ones(3)).transpose())
comTask.setKd(2.0 * np.sqrt(kp_com) * np.matrix(np.ones(3)).transpose())
invdyn.addMotionTask(comTask, w_com, 1, 0.0)

postureTask = tsid.TaskJointPosture("task-posture", robot)
postureTask.setKp(kp_posture * np.matrix(np.ones(robot.nv-6)).transpose())
postureTask.setKd(2.0 * np.sqrt(kp_posture) * np.matrix(np.ones(robot.nv-6)).transpose())
invdyn.addMotionTask(postureTask, w_posture, 1, 0.0)

com_ref = Com[0,0].reshape((-1,1))
trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)

q_ref = q[7:].copy()
q_ref[3] = np.pi/4
q_ref[9] = np.pi/4
q_ref[2] = -np.pi/6
q_ref[8] = -np.pi/6

trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)

solver = tsid.SolverHQuadProgFast("qp solver")
solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)

left_foot = RomeoFoot(lf_frame_name, invdyn, robot)
right_foot = RomeoFoot(rf_frame_name, invdyn, robot)

left_foot.add_contact_task(kp=kp_contact, kd=kd_contact, w_forceRef=w_forceRef, mu=mu)
right_foot.add_contact_task(kp=kp_contact, kd=kd_contact, w_forceRef=w_forceRef, mu=mu)

left_foot.add_foot_steps(left_foot_steps)
right_foot.add_foot_steps(right_foot_steps)

## move to the initial configuration
for i in range(20000):

    sampleCom = trajCom.computeNext()
    comTask.setReference(sampleCom)
    samplePosture = trajPosture.computeNext()
    postureTask.setReference(samplePosture)
    HQPData = invdyn.computeProblemData(t, q, v)
    HQPData.print_all()

    sol = solver.solve(HQPData)
    tau = invdyn.getActuatorForces(sol)
    print("tau: ",tau)

    # bullet
    pbwrapper.send_joint_command(tau)
    p.stepSimulation()
    q, v = pbwrapper.get_state()
    t += dt

    print(robot.com(invdyn.data()))
