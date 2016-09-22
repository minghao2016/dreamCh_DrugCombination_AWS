import config, glob
from job2 import params as job2_param_gen
from job4 import params as job4_param_gen


class Generator():
    def __init__(self, j1_type, problem_num):
        self.j1_type = j1_type
        self.problem_num = problem_num

        self.common = [
                'universe = vanilla',
                'should_transfer_files = IF_NEEDED',
                'executable = /usr/bin/python',
                'should_transfer_files = YES',
                'transfer_executable = false']

        self.config = {
                1: config.j1,
                2: config.j2,
                3: config.j3,
                4: config.j4,
                5: config.j5,
                6: config.j6
                }

    def set_round_num(self, round_num):
        self.round_num = round_num

    def set_job_num(self, job_num):
        self.job_num = job_num
        self.__set_log_path__()

    def __set_log_path__(self):
        self.log_paths = [
                'output = log/' + '.'.join([str(self.job_num), '$(Cluster)','out']),
                'log = log/'    + '.'.join([str(self.job_num), '$(Cluster)','log']),
                'error = log/'  + '.'.join([str(self.job_num), '$(Cluster)','err'])
                ]

    def get_submit_content(self, job_num):
        config = self.config[job_num]

        self.set_job_num(job_num)

        inputs = self.rearrange_inputs(config['inputs'])
        outputs = self.rearrange_outputs(config['outputs'])
        arguments = self.rearrange_arguments(config['arguments'])

        submit_form = [
                '\n'.join(self.common),
                'transfer_input_files = ' + ', '.join(inputs),
                #'transfer_output_files = ' + ', '.join(outputs),
                'transfer_output_files = data',
                config['requirements'],
                'arguments = ' + ' '.join(arguments)]

        """
        if job_num != 2 and job_num != 4:
            submit_form.append('\n'.join(self.log_paths))
        """
        ###Write all logs
        submit_form.append('\n'.join(self.log_paths))


        if 'queue_num' in config:
            submit_form.append('queue ' + str(config['queue_num']))
        else:
            submit_form.append('queue')

        return '\n'.join(submit_form)

    def rearrange_inputs(self, inputs):
        result = [inputs[0]]

        for val in inputs[1:]:
            result.append( '/'.join(['data', str(self.round_num), val]))

        return result

    def rearrange_outputs(self, outputs):
        result = list()
        for val in outputs:
            result.append( '/'.join(['data', str(self.round_num), val]))

        return result

    def rearrange_arguments(self, arguments):
        result = [arguments[0]]
        result.append(str(self.round_num))
        result.append(str(self.problem_num))

        if len(arguments) > 1 :
            for val in arguments[1:]:
                result.append(val)

        return result

    def get_submit_content_job2(self, include_features):
        params = job2_param_gen(self.round_num, self.j1_type, include_features)

        contents = list()

        for feature_candidate, c_value, gamma_value, dataset_num in params:
            self.config[2]['arguments'] = ['J2.py', feature_candidate, str(c_value), str(gamma_value), str(dataset_num)]
            contents.append(self.get_submit_content(2))

        return contents

    def get_submit_content_job3(self, include_features):
        params = job2_param_gen(self.round_num, self.j1_type, include_features)

        contents = list()
        features = set()

        for feature_candidate, c_value, gamma_value, dataset_num in params:
            features.add(feature_candidate)
        for feature in features:
            self.config[3]['arguments'] = ['J3.py', feature]
            self.config[3]['inputs'][1] = 'J2condor/result/' + feature
            contents.append(self.get_submit_content(3))

        return contents

    def get_submit_content_job4(self, include_features):
        params = job4_param_gen(self.j1_type, include_features)

        contents = list()

        for p, q in params:
            self.config[4]['arguments'] = ['J4.py', str(p), str(q)]
            contents.append(self.get_submit_content(4))
        return contents

    def get_dagman_content(self, round_num, job4_size=0):
        round_num = str(round_num)

        submit_dir = '/'.join(['submit', round_num])

        jobs = list()
        dependencies = list()

        # job1
        #jobs.append('JOB FIRST ' + submit_dir + '/1.0.submit')

        # job2
        j2_list = list()
        j2_jobs = glob.glob(submit_dir + '/2.*')
        #for job2_idx in range(len(j2_jobs)):

        for job2_idx in range(len(j2_jobs)):
            idx = str(job2_idx)
            jobs.append('JOB SECOND' + idx + ' ' + submit_dir + '/2.' + idx + '.submit')
            j2_list.append('SECOND' + idx)

        #dependencies.append('PARENT FIRST CHILD ' + ' '.join(j2_list))
        # job3
        jobs.append('JOB THIRD ' + submit_dir + '/3.0.submit')
        dependencies.append('PARENT ' + ' '.join(j2_list) + ' CHILD THIRD')
        """
        # job4
        j4_list = list()

        for job4_idx in range(job4_size):
            idx = str(job4_idx)
            jobs.append('JOB FOURTH' + idx + ' ' + submit_dir + '/4.' + idx + '.submit')
            j4_list.append('FOURTH' + idx)
        dependencies.append('PARENT THIRD CHILD ' + ' '.join(j4_list))

        # job2 retry
        job2_retry = ["Retry " + j2 + " 3" for j2 in j2_list]
        job4_retry = ["Retry " + j4 + " 3" for j4 in j4_list]

        # job 5
        jobs.append('JOB FIFTH ' + submit_dir + '/5.0.submit')
        dependencies.append('PARENT ' + ' '.join(j4_list) + ' CHILD FIFTH')

        #job6
        jobs.append('JOB SIXTH ' + submit_dir + '/6.0.submit')
        dependencies.append('PARENT FIFTH  CHILD SIXTH')
        """
        return '\n'.join(['\n'.join(jobs),'\n'.join(dependencies)])

