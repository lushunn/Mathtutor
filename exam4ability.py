import  pickle
import  numpy as np
import  random
from collections import  deque

with open('all_problem_1.pkl', 'rb') as pkl_file:
    problemdata = pickle.load(pkl_file)
#包含信息 'a','b','pid','gid','gname','gnumber','gnumber_1', 'gnumber_2','vname', 'vid','difficulty'
with open('all_video_sort.pkl','rb') as pkl_file:
    videodata=pickle.load(pkl_file)
# 包含信息 vname ，vid，gnumber-1
from IRT_function import Irt2PL,EAPIrt2PLModel#函数库


class Onion_Doc():
    def __init__(self,problemdata,videodata):
        self.problemdata=problemdata #八下所有习题数据
        self.videodata=videodata #八下所有视频数据
        self.video_pool=deque()#将学视频池
        self.tehta=0.5 #学生预估能力值
        self.left_day=10 #剩下天数
    #todo: 组卷逻辑得更改，初始值为平均theta，组将学习视频下的习题

    def do_math_prob(self,theta,b,a=1,c=0):
        '''
        #做对可能性
        :param theta:  能力值 float
        :param b: 题目难度 float
        :param a:  题目区分度 float
        :param c:  题目猜中概率 float
        :return: float 做对可能性
        '''
        return c+(1-c)/(1+np.exp(-1.702*a*(theta-b)))



    def iif(self,theta,b,a=1,c=0):
        '''
        #信息函数
        :param theta:  能力值 float
        :param b: 题目难度 float
        :param a:  题目区分度 float
        :param c:  题目猜中概率 float
        :return:  flaot 题目的信息值
        '''
        return  1.702**2*a**2*(1-c)/(c+np.exp(1.702*a*(theta-b)))*(1+np.exp(-1.702*a*(theta-b)))**2


    def EM_update(self,scores,a,b):
        '''
        EM迭代更新能力值
        '''
        eap = EAPIrt2PLModel(scores, a, b)
        return eap.res

    def get_anb(self,problem_list):
        '''
        #提取题目list的每个题目的难度和区分度：
        :param problem_list:  题目list
        :return: a,b np.array
        '''
        a=np.array(list(map(lambda x: x[0], problem_list)))
        b=np.array(list(map(lambda x:x[1],problem_list)))
        return a,b


    def chapter_video(self,videodata,chapter):
        '''
         #对应章节的视频
        :param videodata: deque
        :param chapter: int
        :return: deque
        '''
        self.video_pool=deque(filter(lambda x:x[-1]==chapter,videodata))
        return deque(filter(lambda x:x[-1]==chapter,videodata))

    # 随机组卷
    def creat_exam(self,chapter):
        res=[]
        for i in [1,2,3,9]:
            hard_problem=set(map(lambda x: x[-1], self.problemdata[chapter][i]))#该节点下的题目难度池
            hard_problem= sorted(hard_problem)
            #hard_problem[:2]+hard_problem[-1:]
            for hard_level in hard_problem:
                tmp=list(filter(lambda x: x[-1] == hard_level, self.problemdata[chapter][i]))
                res.extend(random.sample(tmp,min(len(tmp),2)))
        return  res[:12]

    #根据一条视频安排习题
    def arrange_problem(self,video_tuple,theta):
            #todo:有些习题对应的二级目标下没有对应的视频，当前没有考虑
            video_id=video_tuple[1]
            video_name = video_tuple[0]
            video_chapter = video_tuple[-1]
            video_problem_pool=[]
            for section in self.problemdata[video_chapter].values():
                if section:
                    video_problem_pool.extend(list(filter(lambda x: x[-2] == video_id, section)))

            return sorted(video_problem_pool,key=lambda x:self.iif(theta,x[1],x[0]),reverse=True)[:6]

    #根据一天看过的视频安排习题
    def arrange_problems(self, video_tuples, theta):
        res=[]
        for i in video_tuples:
            res.extend(self.arrange_problem( i, theta))
        return sorted(res, key=lambda x: self.iif(theta, x[1], x[0]),reverse=True)[:6]

    # 安排视频
    def arrange_video(self,video_pool,left_day):
        #零时函数 batch list
        def batch(iter_obj, n=1):
            l=len(iter_obj)
            for i in range(0, l, n):
                yield iter_obj[i:min(i + n, l)]
        arranged_video=deque()
        batch_size=len(video_pool)//left_day
        for video in batch(list(video_pool),batch_size):
            arranged_video.append(video)
        return arranged_video



#class SimLearnEnv(Onion_Doc):
import time
if __name__ == '__main__':
    Onion_Doc=Onion_Doc(problemdata,videodata)
    #第一天 组卷 加视频
    chapter = int(input("输入章节:"))
    exam=Onion_Doc.creat_exam(chapter)
    a, b = Onion_Doc.get_anb(exam)
    Onion_Doc.video_pool=Onion_Doc.chapter_video(Onion_Doc.videodata,chapter)
    exam_score = []
    for i in range(len(exam)):
        print('该题目信息', exam[i])
        ans = int(input("input simulation answer(1 or 0):"))
        exam_score.append(ans)
    exam_score=np.array(exam_score)
    theta_pass=Onion_Doc.EM_update(scores=exam_score,a=a,b=b)
    time.sleep(1.5)
    print('init theta:',theta_pass)
    problem_duplicate=False
    problem_done_time=0
    done_problem=[]
    done_video=[]
    while Onion_Doc.left_day>0:
        print('--------剩下',Onion_Doc.left_day,'天--------')
        if  not problem_duplicate or problem_done_time>=2:
            print("-------观看以下视频----------")
            video=Onion_Doc.arrange_video(Onion_Doc.video_pool,Onion_Doc.left_day)[0]
            print(video)
            done_video.extend(video)
            problem_done_time=0
            problem_duplicate = False
        elif  problem_duplicate:
            print('-----继续做题-----')
            video = Onion_Doc.arrange_video(Onion_Doc.video_pool, Onion_Doc.left_day+1)[0]
        time.sleep(1.5)
        print('---做题了-----')
        problem=Onion_Doc.arrange_problems(video, theta_pass)
        problem_score=[]
        done_problem.extend(problem)
        print('simulate problem done :')
        for i in range(len(problem)):
            print('该题目信息', problem[i])
            ans = int(input("input simulation answer(1 or 0):"))
            problem_score.append(ans)
        problem_score=np.array(problem_score)
        a, b = Onion_Doc.get_anb(problem)
        theta =Onion_Doc.EM_update(scores=problem_score, a=a, b=b)
        problem_done_time+=1
        print('theta',theta)
        if theta-theta_pass>=0:
            #能力增长 看接下来的视频
            print('----能力增长----')
            for v in video:
                Onion_Doc.video_pool.remove(v)
            Onion_Doc.left_day-=1
            theta_pass=theta
        else:
            print('---能力没增长----')
            Onion_Doc.left_day -= 1
            theta_pass = theta
            problem_duplicate=True

        if Onion_Doc.left_day==0:
            print('结束了')

