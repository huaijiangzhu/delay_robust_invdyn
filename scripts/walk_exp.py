#!/usr/bin/env python
# coding: utf-8

# In[1]:

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

# In[3]:


BMODE = p.DIRECT
LOGGING = True
LOCAL = False

STEP_DURATION = 800
N_STEPS = 6
N_SIMULATION = N_STEPS * STEP_DURATION
DOUBLE_SUPPORT_DURATION = 5

DELAY = 4
NS3 = True
SCENARIO = 'building'
N_EXPS = 100

DELAY_START = 0
ONBOARD_WINDOW = 50

PRELANDING = 10
PRELIFTING = 10

ONBOARD_FREQ = 5


# In[4]:

for trial_id in range(N_EXPS):
    # compute delayed indices
    delayed_ids = []
    print "trial id " + str(trial_id)

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
            
    delayed_ids_raw = copy.copy(delayed_ids)


    # In[5]:


    Com = np.load(parentdir+'/data/com.npy')
    foot_steps = np.load(parentdir+'/data/foot_steps.npy')
    left_foot_steps = foot_steps[1::2]
    right_foot_steps = foot_steps[0::2]


    # In[6]:


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

    sigma_q = 1e-3
    sigma_v = 1e-3 # measurement noise


    # In[7]:


    path = parentdir + '/models/romeo'
    robot_urdf = path + '/urdf/romeo.urdf'
    plane_urdf = parentdir + '/models/plane.urdf'
    vector = se3.StdVec_StdString()
    vector.extend(item for item in path)
    robot = tsid.RobotWrapper(robot_urdf, vector, se3.JointModelFreeFlyer(), False)
    pinocchio_robot = se3.buildModelFromUrdf(robot_urdf, se3.JointModelFreeFlyer())


    # In[8]:


    physicsClient = p.connect(BMODE)#or p.DIRECT for non-graphical version
    dt = 1./1000.
    p.setTimeStep(dt)
    p.setGravity(0,0,-9.81)
    robot_id = p.loadURDF(robot_urdf,[0,0,0])
    plane_id = p.loadURDF(plane_urdf,[0,0,-0.88], useFixedBase=1)


    # In[9]:


    joint_names = [n for n in robot.model().names]
    del joint_names[0]
    del joint_names[0]
    endeff_names = ["l_sole", "r_sole"]
    pbwrapper = PinBulletWrapper(robot_id, pinocchio_robot, joint_names, endeff_names)


    # In[10]:


    p.changeDynamics(plane_id, -1, restitution=0.99,
                    lateralFriction=1.0, spinningFriction=1.0, rollingFriction=1.0,
                    contactStiffness=1e6, contactDamping=2e3)
    p.resetJointState(robot_id, 20, np.pi/2)
    p.resetJointState(robot_id, 27, np.pi/2)


    # In[11]:


    q, v = pbwrapper.get_state()
    assert robot.model().existFrame(rf_frame_name)
    assert robot.model().existFrame(lf_frame_name)


    # In[12]:


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


    # In[13]:


    left_foot = RomeoFoot(lf_frame_name, invdyn, robot)
    right_foot = RomeoFoot(rf_frame_name, invdyn, robot)

    left_foot.add_contact_task(kp=kp_contact, kd=kd_contact, w_forceRef=w_forceRef, mu=mu)
    right_foot.add_contact_task(kp=kp_contact, kd=kd_contact, w_forceRef=w_forceRef, mu=mu)

    left_foot.add_foot_steps(left_foot_steps)
    right_foot.add_foot_steps(right_foot_steps)


    # In[14]:


    ## move to the initial configuration
    for i in range(2000):
        
        sampleCom = trajCom.computeNext()
        comTask.setReference(sampleCom)
        samplePosture = trajPosture.computeNext()
        postureTask.setReference(samplePosture)
        HQPData = invdyn.computeProblemData(t, q, v)
        
        sol = solver.solve(HQPData)
        tau = invdyn.getActuatorForces(sol)
                
        # bullet
        pbwrapper.send_joint_command(tau)    
        p.stepSimulation()
        q, v = pbwrapper.get_state()
        t += dt
        
    Com[:,0,2] = robot.com(invdyn.data())[2, 0]


    # In[15]:


    left_foot.add_motion_task(kp=kp_foot_motion, kd=kd_foot_motion, w=w_foot_motion)
    right_foot.add_motion_task(kp=kp_foot_motion, kd=kd_foot_motion, w=w_foot_motion)


    # In[16]:


    feet = [left_foot, right_foot]
    left_foot.lift()
    left_foot.take_the_next_step()
    right_foot.take_the_next_step()

    swing_foot_idx = 0
    support_foot_idx = 1

    landing = STEP_DURATION
    last_landing = STEP_DURATION
    lifting = landing + DOUBLE_SUPPORT_DURATION
    last_lifting = last_landing + DOUBLE_SUPPORT_DURATION


    # In[17]:


    LOG = []
    SOL = []
    step = 0
    success = True

    for i in range(N_SIMULATION):
        
        data = DotMap()
        LOG.append(data)
        
        swing_foot = feet[swing_foot_idx]
        support_foot = feet[support_foot_idx]
        
        # precompute solution for landing
        if LOCAL and i == landing - PRELANDING:
            swing_foot.land(precomputing=True)
            HQPData = invdyn.computeProblemData(t, q + noise_q, v + noise_v)
            sol_landing = solver.solve(HQPData)
            swing_foot.lift(precomputing=True)
            
        # precompute solution for lifting        
        if LOCAL and i ==  lifting - PRELIFTING:
            swing_foot.land(precomputing=True)
            support_foot.lift(precomputing=True)
            HQPData = invdyn.computeProblemData(t, q + noise_q, v + noise_v)
            sol_lifting = solver.solve(HQPData)
            swing_foot.lift(precomputing=True)
            support_foot.land(precomputing=True)    
                
        if i == landing:
            swing_foot.land()
            
        if i == lifting:
            last_landing = landing
            last_lifting = lifting
            landing += STEP_DURATION
            lifting += STEP_DURATION
            support_foot.lift()
            support_foot.take_the_next_step()
            swing_foot_idx, support_foot_idx = support_foot_idx, swing_foot_idx
            
        right_foot.update(dt)
        left_foot.update(dt)
            
        sampleCom = trajCom.computeNext()
        sampleCom.pos(Com[i,0].reshape(-1, 1))
        sampleCom.vel(Com[i,1].reshape(-1, 1))
        sampleCom.acc(Com[i,2].reshape(-1, 1))
        samplePosture = trajPosture.computeNext()
        postureTask.setReference(samplePosture)
        comTask.setReference(sampleCom)
        
        noise_q = sigma_q * np.matrix(np.random.randn(q.shape[0])).T
        noise_q[:7] = 0
        noise_v = sigma_v * np.matrix(np.random.randn(v.shape[0])).T
        noise_v[:6] = 0
        
        fullqp_start = time.time()
        HQPData = invdyn.computeProblemData(t, q + noise_q, v + noise_v)
        sol_fullqp = solver.solve(HQPData)
        data.fullqp_time = time.time() - fullqp_start
        tau = invdyn.getActuatorForces(sol_fullqp)
        dv = invdyn.getAccelerations(sol_fullqp)

        if(sol.status != 0):
            print "QP problem could not be solved! Error code:", sol.status
            success = False
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
        
        # Perform onboard computation around contact switches
        if LOCAL and step * STEP_DURATION - ONBOARD_WINDOW <= i < step * STEP_DURATION + ONBOARD_WINDOW:
            # when on-board, only compute full qp every ONBOARD_FREQ ms
            if i % ONBOARD_FREQ == 0:
                delay = 2 + int(1000 * data.fullqp_time)
                # 2 ms extra delay for local controller (<= 1ms)
                # and one-time pre-computation spread out over the whole window (possible with multithreading)
                for h in range(ONBOARD_FREQ):
                    # check if this delay < network delay
                    if h < delay:
                        delayed_ids[i + h] = np.maximum(delayed_ids[i + h], i - ONBOARD_FREQ)
                    if h >= delay:
                        delayed_ids[i + h] = np.maximum(delayed_ids[i + h], i)                
                    
        delayed_id = delayed_ids[i]
        data.delay = i - delayed_id
                
        if i == step * STEP_DURATION + ONBOARD_WINDOW:
            step += 1

        delayed_data = LOG[delayed_id]
        delayed_sol = SOL[delayed_id]
        
        if LOCAL and landing <= i < lifting and delayed_id < landing:
            delayed_sol = sol_landing
        
        if LOCAL and last_lifting <= i and delayed_id < last_lifting:
            delayed_sol = sol_lifting
        
        if LOCAL:
            local_start = time.time()
            HQPData = invdyn.computeProblemData(t, q + noise_q, v + noise_v)
            sol = solver.solve_local(HQPData, delayed_sol)
            data.local_time = time.time() - local_start
            tau = invdyn.getActuatorForces(sol)
            dv = invdyn.getAccelerations(sol)
        else: 
            solver.compute_slack(HQPData, delayed_sol)
            sol = delayed_sol
            tau = delayed_data.tau_fullqp
            dv = delayed_data.dv_fullqp
            data.local_time = 0
            
        data.slack = sol.slack
        data.tau = tau
        data.dv = dv
        data.activeset = sol.activeSet
        
        # data logging from noiseless measurements
        HQPData = invdyn.computeProblemData(t, q, v)
        data.com.pos = robot.com(invdyn.data())
        data.com.pos_ref = sampleCom.pos()
        data.left_foot.pos = left_foot.pos(t,q,v).translation
        data.left_foot.pos_ref = left_foot.motion_ref.pos()[:3]
        data.right_foot.pos = right_foot.pos(t,q,v).translation
        data.right_foot.pos_ref = right_foot.motion_ref.pos()[:3]
        
        # pinocchio
        v_mean = v + 0.5*dt*dv
        v += dt*dv
        q = se3.integrate(robot.model(), q, dt*v_mean)
        pbwrapper.reset_state(q, v)
        p.stepSimulation()
        t += dt


    # In[18]:


    p.disconnect()
    print "Simulation completed"


    # In[19]:


    exp_data = DotMap()
    file_name = 'walk_'
    data_path = parentdir + '/data/'
    exp_data.success = success
    exp_data.N_SIMULATION = N_SIMULATION
    exp_data.dt = dt
    exp_data.N_STEPS = N_STEPS
    exp_data.ONBOARD_WINDOW = ONBOARD_WINDOW
    exp_data.STEP_DURATION = STEP_DURATION
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
