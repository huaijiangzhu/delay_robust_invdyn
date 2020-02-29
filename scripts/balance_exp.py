#!/usr/bin/env python
# coding: utf-8

import os
import inspect

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
srcdir = parentdir + '/src'
os.sys.path.insert(1, srcdir)

import numpy as np

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

from dotmap import DotMap
import pickle


BMODE = p.DIRECT
LOGGING = True
LOCAL = False

N_SIMULATION = 5000

DELAY = 1
NS3 = True
SCENARIO = 'factory'
N_EXPS = 100

DELAY_START = 0

IMPACT_START = 1350
IMPACT_DURATION = 200
IMPACT_FORCE = [100, 0, 0]

for trial_id in range(N_EXPS):
    print ("Trial ", trial_id)
    # compute delayed indices
    delayed_ids = []

    if NS3:
        ns3_path = parentdir + '/data/ns3/' + SCENARIO + '/delayed_ids/'
        delayed_ids = np.load(ns3_path + str(trial_id) +'.npy')
        
    else:
        for i in range(N_SIMULATION):
            if i >= DELAY_START:
                delayed_id = np.maximum(i-DELAY, DELAY_START)
            else:
                delayed_id = i
            delayed_ids.append(delayed_id)


    # In[5]:


    LINE_WIDTH = 60
    print "".center(LINE_WIDTH,'#')
    print " Test Task Space Inverse Dynamics ".center(LINE_WIDTH, '#')
    print "".center(LINE_WIDTH,'#'), '\n'

    lxp = 0.14                          # foot length in positive x direction
    lxn = 0.077                         # foot length in negative x direction
    lyp = 0.069                         # foot length in positive y direction
    lyn = 0.069                         # foot length in negative y direction
    lz = 0.105                          # foot sole height with respect to ankle joint
    mu = 1.0                            # friction coefficient
    fMin = 5.0                          # minimum normal force
    fMax = 1000.0                       # maximum normal force
    rf_frame_name = "RAnkleRoll"        # right foot frame name
    lf_frame_name = "LAnkleRoll"        # left foot frame name
    contactNormal = np.matrix([0., 0., 1.]).T   # direction of the normal to the contact surface

    w_com = 1.0                     # weight of center of mass task
    w_posture = 1e-3                # weight of joint posture task
    w_forceRef = 1e-5               # weight of force regularization task

    kp_contact = 10.0               # proportional gain of contact constraint
    kp_com = 10.0                   # proportional gain of center of mass task
    kp_posture = 30.0               # proportional gain of joint posture task

    sigma_q = 1e-2
    sigma_v = 1e-2


    path = parentdir + '/models/romeo'
    robot_urdf = path + '/urdf/romeo.urdf'
    plane_urdf = parentdir + '/models/plane.urdf'
    vector = se3.StdVec_StdString()
    vector.extend(item for item in path)
    robot = tsid.RobotWrapper(robot_urdf, vector, se3.JointModelFreeFlyer(), False)
    pinocchio_robot = se3.buildModelFromUrdf(robot_urdf, se3.JointModelFreeFlyer())


    physicsClient = p.connect(BMODE)
    dt = 1e-3
    p.setTimeStep(dt)
    p.setGravity(0,0,-9.81)
    robot_id = p.loadURDF(robot_urdf,[0,0,0])
    plane_id = p.loadURDF(plane_urdf,[0,0,-0.88], useFixedBase=True)


    joint_names = [n for n in robot.model().names]
    del joint_names[0]
    del joint_names[0]
    endeff_names = ["l_sole", "r_sole"]
    pbwrapper = PinBulletWrapper(robot_id, pinocchio_robot, joint_names, endeff_names)



    p.changeDynamics(plane_id, -1, restitution=0.99,
                    lateralFriction=1.0, spinningFriction=1.0, rollingFriction=1.0,
                    contactStiffness=1e6, contactDamping=2e3)
    p.resetJointState(robot_id, 20, np.pi/3)
    p.resetJointState(robot_id, 27, np.pi/3)


    q, v = pbwrapper.get_state()
    assert robot.model().existFrame(rf_frame_name)
    assert robot.model().existFrame(lf_frame_name)


    t = 0.0                         # time
    invdyn = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
    invdyn.computeProblemData(t, q, v)
    data = invdyn.data()
    contact_Point = np.matrix(np.ones((3,4)) * lz)
    contact_Point[0, :] = [-lxn, -lxn, lxp, lxp]
    contact_Point[1, :] = [-lyn, lyp, -lyn, lyp]

    contactRF =tsid.Contact6d("contact_rfoot", robot, rf_frame_name, contact_Point, contactNormal, mu, fMin, fMax, w_forceRef)
    contactRF.setKp(kp_contact * matlib.ones(6).T)
    contactRF.setKd(2.0 * np.sqrt(kp_contact) * matlib.ones(6).T)
    H_rf_ref = robot.position(data, robot.model().getJointId(rf_frame_name))
    contactRF.setReference(H_rf_ref)
    invdyn.addRigidContact(contactRF)

    contactLF =tsid.Contact6d("contact_lfoot", robot, lf_frame_name, contact_Point, contactNormal, mu, fMin, fMax, w_forceRef)
    contactLF.setKp(kp_contact * matlib.ones(6).T)
    contactLF.setKd(2.0 * np.sqrt(kp_contact) * matlib.ones(6).T)
    H_lf_ref = robot.position(data, robot.model().getJointId(lf_frame_name))
    contactLF.setReference(H_lf_ref)
    invdyn.addRigidContact(contactLF)


    comTask = tsid.TaskComEquality("task-com", robot)
    comTask.setKp(kp_com * matlib.ones(3).T)
    comTask.setKd(2.0 * np.sqrt(kp_com) * matlib.ones(3).T)
    invdyn.addMotionTask(comTask, w_com, 1, 0.0)

    postureTask = tsid.TaskJointPosture("task-posture", robot)
    postureTask.setKp(kp_posture * matlib.ones(robot.nv-6).T)
    postureTask.setKd(2.0 * np.sqrt(kp_posture) * matlib.ones(robot.nv-6).T)
    invdyn.addMotionTask(postureTask, w_posture, 1, 0.0)

    com_ref = robot.com(data)
    trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)

    q_ref = q[7:]
    trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)

    solver = tsid.SolverHQuadProgFast("qp solver")
    solver.resize(invdyn.nVar, invdyn.nEq, invdyn.nIn)


    for i in range(2000):
        
        sampleCom = trajCom.computeNext()
        comTask.setReference(sampleCom)
        samplePosture = trajPosture.computeNext()
        postureTask.setReference(samplePosture)
        HQPData = invdyn.computeProblemData(t, q, v)
        
        sol = solver.solve(HQPData)
        tau = invdyn.getActuatorForces(sol)
        dv = invdyn.getAccelerations(sol)  
            
        if(sol.status!=0):
            print "QP problem could not be solved! Error code:", sol.status
            break
                
        # bullet
        pbwrapper.send_joint_command(tau)    
        p.stepSimulation()
        q, v = pbwrapper.get_state()
        t += dt
        
    com_ref = robot.com(invdyn.data())
    trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)


    LOG = []
    SOL = []
    success = True

    for i in range(0, N_SIMULATION):
        
        data = DotMap()
        LOG.append(data)
        sampleCom = trajCom.computeNext()
        comTask.setReference(sampleCom)
        samplePosture = trajPosture.computeNext()
        postureTask.setReference(samplePosture)
        
        noise_q = sigma_q * np.matrix(np.random.randn(q.shape[0])).T
        noise_q[:7] = 0

        noise_v = sigma_v * np.matrix(np.random.randn(v.shape[0])).T
        noise_v[:6] = 0

        fullqp_start = time.time()
        HQPData = invdyn.computeProblemData(t, q+noise_q, v+noise_v)
        sol_fullqp = solver.solve(HQPData)
        data.fullqp_time = time.time() - fullqp_start
        tau = invdyn.getActuatorForces(sol_fullqp)
        dv = invdyn.getAccelerations(sol_fullqp)

        if(sol_fullqp.status!=0):
            print "QP problem could not be solved! Error code:", sol.status
            break 
        if(np.linalg.norm(dv[:6]) >= 1e3):
            print "Slipped and fell."
            success = False
            break  
        
        SOL.append(sol_fullqp)
        data.q = q + noise_q
        data.v = v + noise_v
        data.noise_q = noise_q
        data.noise_v = noise_v
        
        data.tau_fullqp = tau
        data.dv_fullqp = dv
        data.activeset_fullqp = sol_fullqp.activeSet

        delayed_id = delayed_ids[i]
        data.delay = i - delayed_id
        delayed_data = LOG[delayed_id]
        delayed_sol = SOL[delayed_id]

        if LOCAL:
            local_start = time.time()
            HQPData = invdyn.computeProblemData(t, q + noise_q, v + noise_v)
            sol = solver.solve_local(HQPData, delayed_sol)
            data.local_time = time.time() - local_start
            tau = invdyn.getActuatorForces(sol)

        else:
            # this will change the original data, better copy it
            # but pickling of sol is not supported
            solver.compute_slack(HQPData, delayed_sol)
            sol = delayed_sol
            tau = delayed_data.tau_fullqp
            data.local_time = 0
            
        data.slack = sol.slack
        data.tau = tau
        data.dv = dv
        data.activeset= sol.activeSet
        
        # data logging from noiseless measurements
        HQPData = invdyn.computeProblemData(t, q, v)
        data.com.pos = robot.com(invdyn.data())
        data.com.pos_ref = sampleCom.pos()

        # bullet
        pbwrapper.send_joint_command(tau)
        if IMPACT_START <= i <= IMPACT_START + IMPACT_DURATION:
            p.applyExternalForce(robot_id, 0, IMPACT_FORCE, [0,0,0], p.LINK_FRAME)
        p.stepSimulation()
        q, v = pbwrapper.get_state()
        t += dt


    p.disconnect()
    print "Simulation completed"


    exp_data = DotMap()
    file_name = 'balance_'
    data_path = parentdir + '/data/'
    exp_data.success = success
    exp_data.N_SIMULATION = N_SIMULATION
    exp_data.dt = dt
    exp_data.NS3 = NS3

    if NS3:
        exp_data.scenario = SCENARIO
        file_name = file_name + SCENARIO + str(trial_id)
        data_path += 'ns3/exp_data/'
    else:
        exp_data.delay = DELAY
        exp_data.delay_start = DELAY_START
        file_name += str(DELAY)
        data_path += 'const/exp_data/'

    exp_data.local = LOCAL

    if LOCAL:
        file_name += '_local'
    exp_data.log = LOG

    if LOGGING:
        with open(data_path + file_name + '.pkl', 'wb') as f:
            pickle.dump(exp_data, f, pickle.HIGHEST_PROTOCOL)


    # In[17]:


    if LOGGING:
        with open(data_path + file_name + '.pkl', 'wb') as f:
            pickle.dump(exp_data, f, pickle.HIGHEST_PROTOCOL)


