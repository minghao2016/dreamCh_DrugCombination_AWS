import re, time, os, shutil, sys
import email_sender
from subprocess import check_output, call
from JobGenerator.generator import Generator as Job_Generator
from jobs import J1
from jobs import svr as testset_svr

class Supervisor():
    def __init__(self, round_num, j1_type, problem_num):
        self.j1_type = j1_type
        self.round_num = int(round_num)
        self.problem_num = problem_num

        self.command = {
                'condor_submit': ['condor_submit'],
                'condor_submit_dag': ['condor_submit_dag'],
                'condor_q': {
                    'run':  ['condor_q', '-global', '-run'],
                    'held': ['condor_q', '-global', '-held']
                }
        }

        self.job_gen = Job_Generator(j1_type, problem_num)
        self.run_flag = False

    """
    Check if job(s) exists in Condor Queue
    """
    def exist_jobs(self, q_type):
        jobs = check_output(self.command['condor_q'][q_type])

        #lines = jobs.strip().split('\n')

        if 'ubuntu' in jobs:
            return True
        else:
            return False

    """
    Execute new cycle
    """
    def execute_new_cycle(self):

        # Step1 :
        """
        if self.round_num != 0:
            self.move_J1data()
        """
        if self.round_num != 0:
            self.eval_testset()

        # execute J1
        removed_feature_indexes = self.execute_job1_distribute_data()
        # send previous baseline and remove feature cnt
        self.send_result2email(len(removed_feature_indexes))

        # Step2 : Generate new condor submit according to target dir path
        submit_file_path = self.generate_submit_form(removed_feature_indexes)

        # Step3 : submit condor job submit file
        print "#Execute Problem1." + self.j1_type
        print "#Round " + str(self.round_num)

        self.send_submit_form(submit_file_path)
        self.round_num = self.round_num + 1
	time.sleep(10)

    ################################3
    ## Eval in Testset
    def get_excluded_list(self):
        target_path = 'excludedIndexes.txt'
        f = open(target_path)
        excluded_list = [ int(v.strip()) for v in f]
        f.close()

        return excluded_list

    def get_round_bestParams(self):
        target_path = '/'.join(['data', str(self.round_num-1), 'J3condor', 'result', 'parameter.csv'])
        f = open(target_path)
        lines = f.read().split('\n')

        target_line = lines[1].strip()
        params = target_line.split(',')

        return (float(params[0]), float(params[1]))

    def eval_testset(self):
        (C, gamma) = self.get_round_bestParams()
        excluded_list = self.get_excluded_list()

        print "## Evaluate Test set"
        print (C, gamma)
        print excluded_list

        testset_svr.run(
                '/'.join(['data_1a', 'test', 'data']),
                '/'.join(['data_1a', 'test', 'answers', 'ch1_newtest_excluded.csv']),
                '/'.join(['data', str(self.round_num-1), 'J3condor']),
                C, gamma,
                excluded_list,
                self.problem_num)

    def execute_job1_distribute_data(self):
        #step1 : execute Job on locally
        print "job1 execute"
        removed_feature_indexes = J1.execute(self.round_num)
        print "job1 finish"

        #step2 : distributing data
        #call(['bash', 'uploader.sh', str(self.round_num)])

        return removed_feature_indexes

    def move_J1data(self):
        src_dir = '/'.join(['data', str(self.round_num-1), 'J1condor'])
        dst_dir = '/'.join(['data', str(self.round_num), 'J1condor'])
        shutil.copytree(src_dir, dst_dir)

    def generate_submit_form(self, removed_feature_indexes):
        self.job_gen.set_round_num(self.round_num)

        target_dir = '/'.join(['submit', str(self.round_num)])
        os.makedirs(target_dir)

        # TODO: clean below dirty code
        for job_num in range(2,7):
            if job_num == 2:
                submit_content = self.job_gen.get_submit_content_job2(len(removed_feature_indexes))
            elif job_num == 4:
                submit_content = self.job_gen.get_submit_content_job4(removed_feature_indexes)
                job4_size = len(submit_content)
            else:
                submit_content = [self.job_gen.get_submit_content(job_num)]

            for proc_num in range(len(submit_content)):
                f = open(
                    '/'.join([target_dir,
                    '.'.join([str(job_num),str(proc_num),'submit'])]), 'a' )
                f.write(submit_content[proc_num])
                f.close()

        dag_content = self.job_gen.get_dagman_content(self.round_num, job4_size)
        f = open('/'.join([target_dir, 'DAGman.' + str(self.round_num) + '.dag']), 'a')
        f.write(dag_content)
        f.close()

        return target_dir

    def send_submit_form(self, submit_dir_path):
        call(['condor_submit_dag','-maxidle','1000', submit_dir_path + '/DAGman.' + str(self.round_num) + '.dag'])

        time.sleep(60)

    def send_result2email(self, removed_feature_cnt):
        if self.round_num != 0:

            content = [
                    ':: ' + self.problem_num + '.' + self.j1_type,
                    '\n#Round ' + str(self.round_num-1)]

            f = open('/'.join(['data', str(self.round_num-1), 'J3condor', 'result', 'parameter.csv']))
            content.append(f.read())
            f.close()
            content.append("\n")

            # CV Set
            content.append("## CV Set")
            value_sum = .0
            f = open('/'.join(['data', str(self.round_num-1), 'J3condor', 'result', 'baseline.csv']))
            for line in f :
                if 'index' in line:
                    continue

                line = line.strip()
                value_sum += float(line.split(',')[1])
                content.append(line)
            f.close()

            content.append("# Avg ")
            content.append(str(value_sum/10))

            #Test Set
            content.append("\n\n#Test Set")
            f = open('/'.join(['data', str(self.round_num-1), 'J3condor', 'final_result.txt']))
            content.append(f.read())
            f.close()

            content.append('removed feature cnt: ' + str(removed_feature_cnt) + ' in next Round')
            email_sender.send('\n'.join(content))



if __name__ == '__main__':
    # round number, data set type, problem num
    supervisor = Supervisor(sys.argv[1], sys.argv[2], sys.argv[3])
    while True:
        if supervisor.exist_jobs('run'):
            #TODO: nothing to do
            pass
        elif supervisor.exist_jobs('held'):
            email_sender.send("Job held!!!")
            sys.exit()
        else:
            supervisor.execute_new_cycle()
            time.sleep(10)
