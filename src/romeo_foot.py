from min_jerk_traj import Contact, PolynominalList, constant_poly, poly_points
import tsid
import numpy as np
class RomeoFoot(object):
    def __init__(self, name, invdyn, robot):
        self.name = name
        self.invdyn = invdyn
        self.robot = robot
        self.frame_id = robot.model().getFrameId(name)
        self.in_contact = False
        
    def add_foot_steps(self, foot_steps, step_duration=0.8):
        self.foot_steps = foot_steps
        self.step_duration = step_duration
        self.step_cnt = 0
        self.step_t = 0
        
    def add_motion_task(self, kp=50, kd=1000, w=1.0):
        pos = self.robot.position(self.invdyn.data(), self.robot.model().getJointId(self.name))
        self.motion_traj = tsid.TrajectorySE3Constant(self.name + "_traj", pos)
        self.motion_ref = self.motion_traj.computeNext()
        
        self.motion_task = tsid.TaskSE3Equality(self.name + "_motion", self.robot, self.name)
        self.motion_task.setKp(kp * np.matrix(np.ones(6)).transpose())
        self.motion_task.setKp(kd * np.matrix(np.ones(6)).transpose())
        self.invdyn.addMotionTask(self.motion_task, w, 1, 0.0)
        
    def add_contact_task(self, kp=10, kd=7, w_forceRef=1e-5, mu=0.5):
    
        fMin = 5.
        fMax = 1000.
        
        lxp = 0.14                          # foot length in positive x direction
        lxn = 0.077                         # foot length in negative x direction
        lyp = 0.069                         # foot length in positive y direction
        lyn = 0.069                         # foot length in negative y direction
        lz = 0.105                          # foot sole height with respect to ankle joint
        
        contact_Point = np.matrix(np.ones((3,4)) * lz)
        contact_Point[0, :] = [-lxn, -lxn, lxp, lxp]
        contact_Point[1, :] = [-lyn, lyp, -lyn, lyp]
        contactNormal = np.matrix([0., 0., 1.]).T  

        pos = self.robot.position(self.invdyn.data(), self.robot.model().getJointId(self.name))
        self.contact = tsid.Contact6d(self.name + "_contact", 
                                      self.robot, self.name, 
                                      contact_Point, 
                                      contactNormal, 
                                      mu, fMin, fMax, w_forceRef)
                
        self.contact.setKp(kp * np.matrix(np.ones(6)).transpose())
        self.contact.setKd(kd * np.matrix(np.ones(6)).transpose())
        self.contact.setReference(pos)
        self.invdyn.addRigidContact(self.contact)
        self.in_contact = True

    def take_the_next_step(self, height=0.05, ground_level=-0.812):
        poly_traj = [PolynominalList(), PolynominalList(), PolynominalList()]
        interval = [0, self.step_duration]

        pos = self.robot.position(self.invdyn.data(), self.robot.model().getJointId(self.name))
        init_pos = np.array(pos.translation).squeeze()

        end_pos = np.zeros(3)
        end_pos[:2] = self.foot_steps[self.step_cnt][:2]
        end_pos[2] = ground_level
        init_pos[1] = self.foot_steps[self.step_cnt][1]
        init_pos[2] = ground_level 

        for idx in range(3):
            via = None
            
            if idx == 2:
                via = height + ground_level
                
            poly = poly_points(interval, 
                               init_pos[idx], 
                               end_pos[idx], via)
            poly_traj[idx].append(interval, poly)
            
        self.motion_poly = poly_traj
        self.step_cnt += 1
        
    def update(self, dt):

        motion_poly = self.motion_poly
        motion_ref = self.motion_ref
        pos = motion_ref.pos()
        vel = motion_ref.vel()
        acc = motion_ref.acc()
        
        for idx in range(3):
            pos[idx] = motion_poly[idx].eval(self.step_t)
            vel[idx] = motion_poly[idx].deval(self.step_t)
            acc[idx] = motion_poly[idx].ddeval(self.step_t)
            
        motion_ref.pos(pos)
        motion_ref.vel(vel)
        motion_ref.acc(acc)
        
        self.motion_task.setReference(motion_ref)
        self.motion_ref = motion_ref
        
        if (not self.in_contact) and self.step_t<=self.step_duration:
            self.step_t += dt
            
    def land(self, precomputing=False):
        if not self.in_contact:
            pos = self.robot.position(self.invdyn.data(), self.robot.model().getJointId(self.name))
            pos.translation = self.motion_ref.pos()[:3]
            self.contact.setReference(pos)
            self.invdyn.addRigidContact(self.contact)
            self.in_contact = True
        
    def lift(self, precomputing=False):
        if self.in_contact:
            self.invdyn.removeRigidContact(self.contact.name, 0)
            if not precomputing:
                self.step_t = 0
            self.in_contact = False

    def pos(self, t, q, v):
        HQPData = self.invdyn.computeProblemData(t, q, v)
        pos = self.robot.position(self.invdyn.data(), self.robot.model().getJointId(self.name))
        return pos 