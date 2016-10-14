import re, time, os, shutil, sys
import email_sender
from pprint import pprint
from subprocess import check_output, call
from JobGenerator.generator import Generator as Job_Generator
from jobs import J1
from jobs import svr as testset_svr
import glob
import pandas as pd
from operator import itemgetter

class Supervisor():
    def __init__(self, round_num, j1_type='a', problem_num='a'):
        self.j1_type = j1_type
        self.round_num = int(round_num)
        self.problem_num = problem_num

        self.command = {
                'condor_submit': ['condor_submit'],
                'condor_submit_dag': ['condor_submit_dag'],
                'condor_q': {
                    'run':  ['condor_q'],
                    'held': ['condor_q', '-held']
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
        if 'condor_dagman' in jobs:
            return True
        else:
            return False

    """
    Execute new cycle
    """
    def execute_new_cycle(self):
        if self.round_num != 0:
            print "## Previous round result"
            try:
                os.makedirs('/'.join(['data', str(self.round_num-1), 'J6condor', 'result']))
            except:
                pass

            result_included_features = dict()

            base_dir = '/'.join(['data', str(self.round_num-1), 'J3condor', 'result']) + '/'
            for feature_dir in glob.glob(base_dir+'**'):
                feature = feature_dir.split('/')[-1]
                df = pd.read_csv('/'.join([feature_dir, 'baseline.csv']))
                mean = df.baseline.mean()
                df = pd.read_csv('/'.join([feature_dir, 'parameter.csv']))
                C = str(df.C[0])
                gamma = str(df.Gamma[0])

                result_included_features[(feature, C, gamma)] = mean

            sorted_test = sorted(result_included_features.items(), key=itemgetter(1), reverse=True)
            pprint(sorted_test)
            # select top1 feature
            best_test = sorted_test[0][0]
            f = open('/'.join(['data', str(self.round_num-1), 'J6condor', 'result', 'add_feature.txt']), 'w')
            f.write(best_test[0])
            f.close()

            shutil.copy('/'.join([base_dir,best_test[0], 'parameter.csv']),'/'.join(['data', str(self.round_num-1), 'J6condor', 'parameter.csv']))


        # execute J1
        #removed_feature_indexes = self.execute_job1_distribute_data()
        default_features  = J1.execute(self.round_num)
        if self.round_num != 0:
            self.send_result2email(sorted_test)

        # Step2 : Generate new condor submit according to target dir path
        submit_file_path = self.generate_submit_form(default_features)
        # remove previous logs
        shutil.rmtree('log/')
        os.makedirs('log')
        # Step3 : submit condor job submit file
        print "#Execute Problem1." + self.j1_type
        print "#Round " + str(self.round_num)

        self.send_submit_form(submit_file_path)
        self.round_num = self.round_num + 1
	time.sleep(10)

    ################################3
    ## Eval in Testset
    """
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
    """

    def move_J1data(self):
        src_dir = '/'.join(['data', str(self.round_num-1), 'J1condor'])
        dst_dir = '/'.join(['data', str(self.round_num), 'J1condor'])
        shutil.copytree(src_dir, dst_dir)

    def generate_submit_form(self, include_features):
        self.job_gen.set_round_num(self.round_num)

        target_dir = '/'.join(['submit', str(self.round_num)])
        os.makedirs(target_dir)

        for job_num in range(2,4):
            if job_num == 2:
                submit_content = self.job_gen.get_submit_content_job2(include_features)
            elif job_num == 3:
                submit_content = self.job_gen.get_submit_content_job3(include_features)
            elif job_num == 4:
                submit_content = self.job_gen.get_submit_content_job4(include_features)
                job4_size = len(submit_content)
            else:
                submit_content = [self.job_gen.get_submit_content(job_num)]

            for proc_num in range(len(submit_content)):
                f = open(
                    '/'.join([target_dir,
                    '.'.join([str(job_num),str(proc_num),'submit'])]), 'a' )
                f.write(submit_content[proc_num])
                f.close()

        job4_size = 0
        dag_content = self.job_gen.get_dagman_content(self.round_num, job4_size)
        f = open('/'.join([target_dir, 'DAGman.' + str(self.round_num) + '.dag']), 'a')
        f.write(dag_content)
        f.close()

        return target_dir

    def send_submit_form(self, submit_dir_path):
        call(['condor_submit_dag','-maxidle','1000', submit_dir_path + '/DAGman.' + str(self.round_num) + '.dag'])

        time.sleep(60)

    def send_result2email(self, sorted_test):
        if self.round_num != 0:

            content = [
                    '############################################',
                    '## Round ' + str(self.round_num-1),
                    '############################################']

            base_dir = '/'.join(['data', str(self.round_num-1)])
            # default feature group
            f = open('/'.join([base_dir, 'default_groups.txt']))

            content.append('\n######################\n#  Default Feature\n######################s')
            content.append('# Groups')
            content.append(f.read().strip())
            f.close()
            # default feature cnt
            f = open('/'.join([base_dir, 'default_ids.txt']))
            content.append('# cnt')
            content.append(str(len(f.read().strip().split('\n'))))
            f.close()

            content.append('\n######################\n#  Selected Feature\n######################s')
            # new feature gruop
            f = open('/'.join([base_dir, 'J6condor', 'result', 'add_feature.txt']))
            content.append('# Group')
            content.append(f.read())
            f.close()

            # parameter
            f = open('/'.join([base_dir, 'J6condor', 'parameter.csv']))
            content.append('# New Parameter')
            content.append(f.read().strip())
            f.close()

            # round detail result
            content.append('\n######################\n#  Detail Result\n######################s')
            for t in sorted_test:
                ((feature, C, gamma), score) = t
                content.append('{score}\t{c}\t{gamma}\t{feature}'.format(
                    feature=feature, c=C, gamma=gamma, score=score))
            print '\n'.join(content)
            email_sender.send('\n'.join(content))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-round', type=int, required=False, default=0)
    parser.add_argument('-ch', type=int, required=True,
            help='challenge number: 1 or 2')
    args = parser.parse_args()

    start_round = args.round
    problem_num = args.ch

    print '-------------------------------'
    print 'Start Round: {0}\nCh_num: {1}'.format(start_round, problem_num)
    print '-------------------------------'
    supervisor = Supervisor(start_round, problem_num=problem_num)

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
