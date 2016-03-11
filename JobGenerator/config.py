j1 = {
    'inputs': ['jobs/J1.py, jobs/Constant.py, jobs/uploader.sh, jobs/remote_cmd.sh', 'J6condor', 'J1condor'],
    'outputs': [],
    'arguments': ['J1.py'],
    'requirements': 'Requirements = (OpSys == "LINUX") && (machine == "power11")'
}

j2 = {
    'inputs': ['jobs/J2.py', 'J1condor'],
    'outputs': [],
    'arguments': ['J2.py'],
    'requirements': 'Requirements = (OpSys == "LINUX")'
}

j3 = {
    'inputs': ['jobs/J3.py, excludedTestSet', 'J2condor', ],
    'outputs': [],
    'arguments': ['J3.py'],
    'requirements': 'Requirements = (OpSys == "LINUX")'

}

j4 = {
    'inputs': ['jobs/J4.py', 'J1condor', 'J3condor', 'J6condor'],
    'outputs': [],
    'arguments': ['J4.py'],
    'requirements': 'Requirements = (OpSys == "LINUX")'
}

j5 = {
    'inputs': ['jobs/J5.py,excludedTestSet', 'J3condor', 'J4condor'],
    'outputs': [],
    'arguments': ['J5.py'],
    'requirements': 'Requirements = (OpSys == "LINUX")'
}

j6 = {
    'inputs': ['jobs/J6.py', 'J5condor'],
    'outputs': [],
    'arguments': ['J6.py'],
    'requirements': 'Requirements = (OpSys == "LINUX")'
}
