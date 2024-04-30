"""
Copyright (c) 2024, Intel Corporation
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
*  Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR APARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHTOWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOTLIMITEDTO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANYTHEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF -THE USEOF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import csv
from pyVim.connect import SmartConnect, Disconnect
from pyVmomi import vim, vmodl
from tools import tasks
import atexit
import getpass
import csv

CSV_HEADERS = ["vm_name", "start", "stride", "force"]


def strtobool(txt):
    return txt.lower() in ["true", "1", "y", "yes", "t"]


class VMConfig:
    def __init__(self, vm_name, start, stride, force):
        self.vm_name = vm_name
        self.start = start
        self.stride = stride
        self.force = force


def generate_from_csv(csv_file):
    config_list = []
    with open(csv_file, newline="") as csvf:
        reader = csv.DictReader(csvf, fieldnames=CSV_HEADERS)
        for row in reader:
            config_list.append(
                VMConfig(
                    row[CSV_HEADERS[0]],
                    int(row[CSV_HEADERS[1]]),
                    int(row[CSV_HEADERS[2]]),
                    strtobool(row[CSV_HEADERS[3]]),
                )
            )
    return config_list


def generate_from_args(args):
    config_list = []
    cfg = VMConfig(args.vm_name, args.start, args.stride, args.force)
    if args.no_pin:
        cfg.stride = 0
    elif args.no_ht:
        cfg.stride = 1

    config_list.append(cfg)
    return config_list


def connect_to_vc(host, user, pwd, disableSslCertValidation=False):
    si = None
    try:
        si = SmartConnect(
            host=host,
            user=user,
            pwd=pwd,
            disableSslCertValidation=disableSslCertValidation,
        )

        atexit.register(Disconnect, si)
    except Exception as error:
        print(error)

    return si


def disconnect_vc(si):
    si.Disconnect()


def get_vm(vm_name, si):
    content = si.RetrieveContent()
    vm = None
    container = content.viewManager.CreateContainerView(
        content.rootFolder, [vim.VirtualMachine], True
    )

    for obj in container.view:
        if obj.name == vm_name:
            vm = obj

    container.Destroy()
    return vm


def power_on_VM(si, vm):
    state = vm.runtime.powerState
    if state == "poweredOn":
        return

    print("Starting VM from " + state)
    task = vm.PowerOn()
    tasks.wait_for_tasks(si, [task])


def power_off_VM(si, vm, force=False):
    state = vm.runtime.powerState
    if state == "poweredOff":
        return True

    if not force:
        print("Need to shut down VM to proceed. Press Y to continue")
        inp = input()
        inp = inp.strip().lower()
        if inp not in ["y", "yes"]:
            return False

    print("Shutting down VM from " + state)
    task = vm.PowerOff()
    tasks.wait_for_tasks(si, [task])
    return True


def update_config(si, vm, extraConfig):
    config_spec = vim.vm.ConfigSpec(extraConfig=extraConfig)
    task = vm.ReconfigVM_Task(config_spec)
    tasks.wait_for_tasks(si, [task])
    print("VM re-configured")


def remove_old_pinning(extraConfig):
    newConfig = []
    for cfg in extraConfig:
        if "sched.vcpu" in cfg.key or "sched.cpu.affinity.exclusive" in cfg.key:
            cfg.value = ""

        newConfig.append(cfg)

    return newConfig


def print_pinning(extraConfig):
    newConfig = []
    strConfig = ""
    for cfg in extraConfig:
        if "sched.vcpu" in cfg.key or "sched.cpu.affinity.exclusive" in cfg.key:
            strConfig += "{}:{}; ".format(cfg.key, cfg.value)

    print(strConfig)

    return newConfig


def generate_pinned_config(extraConfig, start, cores, stride, num_numas, numa_size):
    ind = 0
    cores_per_numa = int(cores / num_numas)
    for i in range(num_numas):
        st = start + i * numa_size
        for j in range(st, st + cores_per_numa * stride, stride):
            cfg = vim.option.OptionValue()
            cfg.key = "sched.vcpu{}.affinity".format(ind)
            core_to_pin = j
            cfg.value = "{}".format(core_to_pin)
            ind += 1
            extraConfig.append(cfg)

    cfg = vim.option.OptionValue()
    cfg.key = "sched.cpu.affinity.exclusive"
    cfg.value = "TRUE"
    ind += 1
    extraConfig.append(cfg)
    return extraConfig


def generate_txt_config(args):
    f = open(args.out, "w")
    for i in range(args.start, args.cores):
        f.write('sched.vcpu{}.affinity="{}"\n'.format(i, i * 2))
    f.close()
