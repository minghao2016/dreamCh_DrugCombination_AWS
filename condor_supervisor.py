import re, time, os, shutil, sys
import email_sender
from subprocess import check_output, call
from JobGenerator.generator import Generator as Job_Generator
from jobs import J1

class Supervisor():
    def __init__(self):
        self.round_num = 0
        self.command = {
                'condor_submit': ['condor_submit'],
                'condor_submit_dag': ['condor_submit_dag'],
                'condor_q': {
                    'run':  ['condor_q', '-global', '-run'],
                    'held': ['condor_q', '-global', '-held']
                }
        }

        self.job_gen = Job_Generator()
        self.run_flag = False

    """
    Check if job(s) exists in Condor Queue
    """
    def exist_jobs(self, q_type):
        jobs = check_output(self.command['condor_q'][q_type])

        lines = jobs.strip().split('\n')
        if len(lines) > 1:
            return True
        else:
            return False

    """
    Execute new cycle
    """
    def execute_new_cycle(self):
        # Step0: send previous result
        self.send_result2email()

        # Step1 :
        if self.round_num != 0:
            self.move_J1data()

        self.execute_job1_distribute_data()
        sys.exit()
        # Step2 : Generate new condor submit according to target dir path
        submit_file_path = self.generate_submit_form()

        # Step3 : submit condor job submit file
        self.send_submit_form(submit_file_path)
        self.round_num += 1

    def execute_job1_distribute_data(self):
        #step1 : execute Job on locally
        if self.round_num != 0:
            J1.execute(self.round_num)

        #step2 : distributing data
        call(['bash', 'uploader.sh', str(self.round_num)])

    def move_J1data(self):
        src_dir = '/'.join(['data', str(self.round_num-1), 'J1condor'])
        dst_dir = '/'.join(['data', str(self.round_num), 'J1condor'])
        shutil.copytree(src_dir, dst_dir)

    def move_J6data(self):
        src_dir = '/'.join(['data', str(self.round_num-1), 'J6condor'])
        dst_dir = '/'.join(['data', str(self.round_num), 'J6condor'])
        shutil.copytree(src_dir, dst_dir)

    def generate_submit_form(self):
        self.job_gen.set_round_num(self.round_num)

        target_dir = '/'.join(['submit', str(self.round_num)])
        os.makedirs(target_dir)

        # TODO: clean below dirty code
        for job_num in range(2,7):
            if job_num == 2:
                submit_content = self.job_gen.get_submit_content_job2()
            elif job_num == 4:
                submit_content = self.job_gen.get_submit_content_job4()
            else:
                submit_content = [self.job_gen.get_submit_content(job_num)]

            for proc_num in range(len(submit_content)):
                f = open(
                    '/'.join([target_dir,
                    '.'.join([str(job_num),str(proc_num),'submit'])]), 'a' )
                f.write(submit_content[proc_num])
                f.close()

        dag_content = self.job_gen.get_dagman_content(self.round_num)
        f = open('/'.join([target_dir, 'DAGman.' + str(self.round_num) + '.dag']), 'a')
        f.write(dag_content)
        f.close()

        return target_dir

    def send_submit_form(self, submit_dir_path):
        call(['condor_submit_dag', '-maxidle','1000',submit_dir_path + '/DAGman.' + str(self.round_num) + '.dag'])

        time.sleep(20)
        self.run_flag = True

    def send_result2email(self):
        if self.round_num != 0:
            content = ['#Round ' + str(self.round_num-1)]

            f = open('/'.join(['data', str(self.round_num-1), 'J3condor', 'result', 'baseline.csv']))
            content.append(f.read())

            f.close()

            email_sender.send('\n\n'.join(content))



if __name__ == '__main__':
    supervisor = Supervisor()

    while True:
        if len(sys.argv) > 1:
            supervisor.round_num = int(sys.argv[1])

        if supervisor.exist_jobs('run'):
            #TODO: nothing to do
            pass
        elif supervisor.exist_jobs('held'):
            #TODO: save log and send email
            pass
        else:
            supervisor.execute_new_cycle()
            time.sleep(2)

    #supervisor.generate_submit_form()
