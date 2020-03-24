

class Attention():
    """
    关于agent观测的理解：
        每个agent观测输出是一个150维的向量，实际上是一个6*5*5的tensor，第一维有6个信息，分别为type(取值为1或者-1)、id、
        health_point、cool(or uncool)、x坐标、y坐标
        第二维和第三维共同构成一个矩阵，表示了该agent的视野范围，原始配置为5*5
        其中在以agent为中心的视野范围内，每个位置标零表示该位置没有单位，该位置的六个信息均为零，即6个5*5矩阵中对应的位置都为0
        存在单位的位置上要么是agent，要么是opponent，type为1表示为友军，-1为敌军；x坐标和y坐标表示观测到的单位在整个地图中的位置(绝对
        位置)
        视野范围的中心位置应该是agent自身的信息
        一个部分可观测的问题
    e.g.
        obs[0][i][j] = 1表示agent的观测范围内(i,j)的位置(相对位置)上存在一个单位，该单位为友军
    根据以上讨论，要将对每个agent的观测信息分开，每个agent的观测信息实际上是一个6维的向量

    """
