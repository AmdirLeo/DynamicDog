import rclpy                                           
from rclpy.node import Node                             
from sensor_msgs.msg import Range      
from geometry_msgs.msg import PoseStamped             
from protocol.msg import MotionServoCmd
import time 
import math


def EulerAndQuaternionTransform( intput_data):
    """
        四元素与欧拉角互换
    """
    data_len = len(intput_data)
    angle_is_not_rad = True
     
 
    w = intput_data[0] 
    x = intput_data[1]
    y = intput_data[2]
    z = intput_data[3]

    r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = math.asin(2 * (w * y - z * x))
    y = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    if angle_is_not_rad : # pi -> 180
        r = math.degrees(r)
        p = math.degrees(p)
        y = math.degrees(y)
    return [r,p,y]


def cal(x0,y0,x,y,base=0,dire=0):#x0y0一般为狗
    """
    计算角度，返回方位，左边为负，右边为正
    """
    dx=x-x0
    dy=y-y0
    if dx==0:
        if dy>0:
            angle= 90
        else:
            angle= -90
    else:
        angle=math.degrees(math.atan(dy/dx))

    if dx<0:
        if angle>0:#三象限
            angle=angle-180
        else:#二象限
            angle=angle+180

    #此时已经得到相对于x轴的角度

    angle+=base
    angle-=dire

    if angle>180:
        angle-=360
    if angle<-180:
        angle+=360
    return angle

#朝向物体
def toward(x):
    vz=0
    if x>12:
        vz=0.4

    elif x>5:
        vz=0.2

    elif x<-12:
        vz=-0.4

    elif x<-5:
        vz=-0.2
    else:
        vz=0.0
    return vz







class UltraNode(Node):
    def __init__(self,name):
        super().__init__(name)
        self.dogname="dogc1"#12-1 NB408 itself

        self.dog2="dogm"#12-1itself
        self.dog1="dogh"#12-2

        self.ball="greenball"

        self.x1=0
        self.x2=0
        self.y1=0
        self.y2=0
        self.bx=0
        self.by=0
        self.dire1=0#direction
        self.dire2=0

        self.angle2=48#towards tv&goal:138 y轴在狗系中138度  48指x轴相对于狗是48度
        self.angle1=-7#y轴在狗系中83度  x轴在狗系中-7度

        self.vx=0.0
        self.vy=0.0
        self.vz=0.0

        self.doorx=0.339#0.589
        self.doory=5.490#-3.490

        
        self.pre_time=time.time()
        self.gua_time=time.time()
        # self.sub=self.create_subscription(Range, f'/{self.dogname}/ultrasonic_payload',self.listener_callback,10)

        self.sub122=self.create_subscription(PoseStamped, f'/vrpn/{self.dog2}/pose',self.listener_callback2,10)
        self.sub121=self.create_subscription(PoseStamped, f'/vrpn/{self.dog1}/pose',self.listener_callback1,10)
        self.subball=self.create_subscription(PoseStamped, f'/vrpn/{self.ball}/pose',self.listener_callbackb,10)

        self.pub = self.create_publisher(MotionServoCmd, f'/{self.dogname}/motion_servo_cmd', 10)                  
        self.cnt=0
        self.finish_time=0


        self.timer = self.create_timer(0.1,self.main_loop)
        

    def listener_callback2(self,msg):#dog2
        self.x2=msg.pose.position.x
        self.y2=msg.pose.position.y
        quaternion = [msg.pose.orientation.x, msg.pose.orientation.y,  
                      msg.pose.orientation.z,  msg.pose.orientation.w]  
 
        # 将四元数转换为欧拉角 
        roll, pitch, yaw = EulerAndQuaternionTransform(quaternion) 
 
        self.dire2=roll
        # self.get_logger().info('122 r:"%d"\n'% roll)

    def listener_callback1(self,msg):#dog1
        self.x1=msg.pose.position.x
        self.y1=msg.pose.position.y
        quaternion = [msg.pose.orientation.x, msg.pose.orientation.y,  
                      msg.pose.orientation.z,  msg.pose.orientation.w]
        roll, pitch, yaw = EulerAndQuaternionTransform(quaternion) 
        self.dire1=roll
        # self.get_logger().info('121 r:"%d"\n'% roll)

    def listener_callbackb(self,msg):
        self.bx=msg.pose.position.x
        self.by=msg.pose.position.y
        # self.get_logger().info('x is : "%.3f" and y is "%.3f"\n'

    def kick(self):
        print('k')
        ba=cal(self.x2,self.y2,self.bx,self.by,self.angle2,self.dire2)#ball angle
        move_msg = MotionServoCmd()  
        move_msg.motion_id = 308                              
        move_msg.cmd_type = 1                                            
        move_msg.value = 2
        move_msg.step_height = [0.05, 0.05]


        # 锁定球的角度  朝向球
        self.vz=toward(ba)

        if self.vz!=0.0:
            self.pre_time=time.time()

        da=cal(self.x2,self.y2,self.doorx,self.doory,self.angle2,self.dire2)#dool angle
        if da>ba+10:
            self.vy=-0.2
            self.pre_time=time.time()
        elif da<ba-10:
            self.vy=0.2
            self.pre_time=time.time()
        else:
            self.vy=0.0

        r=math.sqrt(pow(self.x2-self.bx,2)+pow(self.y2-self.by,2))
        # print(r)

        if r>1.5:
            self.vy=0.0
            self.vx=0.6
            self.pre_time=time.time()
        elif r>0.7:
            self.vx=0.3
            self.pre_time=time.time()
        elif r<0.5:
            self.vx=-0.1
            self.pre_time=time.time()
        else:
            self.vx=0.0

        if(time.time()-self.pre_time>0.15):
            while(time.time()-self.pre_time<2):
                self.vx=1.0
                move_msg.vel_des = [self.vx, 0.0, 0.0]  
                self.pub.publish(move_msg)
                if(self.y2>self.by):
                    break
                if self.y2>4.3:
                    self.vx=-0.1
                    break
            self.pre_time=time.time()
        print(r,self.vx,self.vy,self.vz)
        move_msg.vel_des = [self.vx, self.vy, self.vz]  
        self.pub.publish(move_msg)


    def defence(self):
        print('d')
        move_msg = MotionServoCmd()  
        move_msg.motion_id = 308                              
        move_msg.cmd_type = 1                                            
        move_msg.value = 2
        move_msg.step_height = [0.05, 0.05]
        ha=cal(self.x2,self.y2,self.x2,1000.0,self.angle2,self.dire2)#door angle

        self.vz=toward(ha)
        # print(ha,"  and  ",self.vz)

        if ha>90 or ha<-90:
            self.vx=0.0
        elif ha>45 or ha<-45:
            self.vx=-0.3
        else:
            self.vx=-0.5

        # 后退不能撞到球
        if self.x2<min(self.bx,self.x1)-0.4 or self.x2>max(self.bx,self.x1)+0.4:
            self.vy=0.0
        else:
            if self.x2>max(self.bx,self.x1):
                self.vy=-0.3#往左为正
            else:
                self.vy=0.3



        self.vy=0.0
        move_msg.vel_des = [self.vx, self.vy, self.vz]  
        self.pub.publish(move_msg)

        
        
    def positive(self):#kick+defence
        if(self.y2<self.by+0.100):
            self.kick()
        else:
            if(self.y2<-4.1):
                self.kick()#后退不能出场地
            else:
                self.defence()

    # 策应
    def assist(self):
        print('a')
        move_msg = MotionServoCmd()  
        move_msg.motion_id = 308                              
        move_msg.cmd_type = 1                                            
        move_msg.value = 2
        move_msg.step_height = [0.05, 0.05]

        ha=cal(self.x2,self.y2,self.x2,1000.0,self.angle2,self.dire2)#door angle
        self.vz=toward(ha)

        # 和前狗保持距离
        if self.y2<self.y1-1.8:
            self.vx=0.4
        elif self.y2>self.y1-1.5:
            self.vx=-0.4
        else:
            self.vx=0.0

        # 和球保持水平
        if self.x2>self.bx+0.100:
            self.vy=0.3
        elif self.x2<self.bx-0.100:
            self.vy=-0.3
        else:
            self.vy=0.0

        move_msg.vel_des = [self.vx, self.vy, self.vz]  
        self.pub.publish(move_msg)

    def guard(self):
        print('g')
        move_msg = MotionServoCmd()  
        move_msg.motion_id = 308                              
        move_msg.cmd_type = 1                                            
        move_msg.value = 2
        move_msg.step_height = [0.05, 0.05]

        ha=cal(self.x2,self.y2,self.x2,1000.0,self.angle2,self.dire2)#door angle
        self.vz=toward(ha)

        # 和球保持水平
        if self.x2>self.bx+0.100:
            self.vy=0.3
            self.gua_time=time.time()
        elif self.x2<self.bx-0.100:
            self.vy=-0.3
            self.gua_time=time.time()
        else:
            self.vy=0.0

        # 在门前
        if self.y2<-4.1:
            self.vx=0.2
        elif self.y2>-2.5:
            # self.gua_time=time.time()
            self.vx=-0.7
            if self.x2<self.bx-0.4 or self.x2>self.bx+0.4:
                self.vy=0.0
            else:
                if self.x2>self.bx:
                    self.vy=-0.3#往左为正
                else:
                    self.vy=0.3
        elif self.y2>-4.0:
            # self.gua_time=time.time()
            self.vx=-0.4
        else:
            self.vx=0.0

        r=math.sqrt(pow(self.x2-self.bx,2)+pow(self.y2-self.by,2))
        if(r>0.7):
            self.gua_time=time.time()

        if(time.time()-self.gua_time>0.2):
            while(time.time()-self.gua_time<1.2):
                self.vx=1.0
                move_msg.vel_des = [self.vx, 0.0, 0.0]  
                self.pub.publish(move_msg)
            self.gua_time=time.time()

        move_msg.vel_des = [self.vx, self.vy, self.vz]  
        self.pub.publish(move_msg)

        

    def main_loop(self):
        if self.by>0.0:
            if self.by>self.y1 and self.by>self.y2:
                r2=math.sqrt(pow(self.x2-self.bx,2)+pow(self.y2-self.by,2))
                r1=math.sqrt(pow(self.x1-self.bx,2)+pow(self.y1-self.by,2))
                if(r1>r2):
                    self.positive()
                else:
                    self.assist()
            else:
                self.positive()
        else:
            if self.y2<self.y1:
                self.guard()
            else:
                self.positive()
        return


        
        

        

        




        
        


def main(args=None):
    rclpy.init(args=args)
    node =UltraNode("topic_vr")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


