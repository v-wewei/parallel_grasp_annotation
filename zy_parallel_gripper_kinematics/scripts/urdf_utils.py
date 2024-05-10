from urdfpy import URDF



def reprase_urdf():
    robot = URDF.load('../urdf/speed_hand.urdf')
    robot.save('../urdf/speed_hand2.urdf')


# def vis_urdf():


reprase_urdf()