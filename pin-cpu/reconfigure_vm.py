"""
Copyright (c) 2024, Intel Corporation
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

*  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
*  Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR APARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHTOWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOTLIMITEDTO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANYTHEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF -THE USEOF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from argparse import ArgumentParser
from utils import *


if __name__ == "__main__":
    parser = ArgumentParser(description="Pin a VM")
    parser.add_argument(
        "--host", type=str, default=None, help="vCenter URL. e.g. abc.vcsa.com"
    )
    parser.add_argument(
        "--user",
        type=str,
        default=None,
        help="Privileged user to reconfigure VM. e.g. administrator@vsphere.local",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Password to the account. Alternatively, use stdin when prompted",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print config instead of updating",
    )
    parser.add_argument(
        "--vm-name", type=str, default=None, help="Name of the virtual machine to pin"
    )
    parser.add_argument(
        "--insecure", action="store_true", help="Ignore TLS certificate verification"
    )
    parser.add_argument(
        "--no-pin",
        action="store_true",
        help="Remove pinning from the VM. Equivalent to stride=0",
    )
    parser.add_argument(
        "--no-ht",
        action="store_true",
        help="Remove pinning from the VM. Equivalent to stride=1",
    )
    parser.add_argument(
        "-s", "--start", type=int, default=0, help="Starting CPU/ thread to pin from"
    )
    parser.add_argument(
        "-t",
        "--stride",
        type=int,
        default=2,
        help="Option to control size of stride. Default is 2 (HT on)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Automatically shutdown VM if powered on without prompt",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="CSV file for batch operations",
    )
    parser.add_argument(
        "-n",
        "--num-numas",
        type=int,
        default=1,
        help="Number of numa nodes to use. If specified, it will pin <cores>/<num_numas> per numa starting from <start> in each numa",
    )
    parser.add_argument(
        "-z",
        "--numa-size",
        type=int,
        default=0,
        help="Size of numa node. Needed if num-numas is specified",
    )
    args = parser.parse_args()

    if args.password is None:
        print("Password is needed for the provided user account {}".format(args.user))
        args.password = getpass.getpass()

    si = connect_to_vc(args.host, args.user, args.password, args.insecure)
    if si is None:
        exit()

    print("Established session with vCenter {} sucessfully".format(args.host))
    vm_cfg = None
    if args.input is not None:
        vm_cfg = generate_from_csv(args.input)
    else:
        vm_cfg = generate_from_args(args)

    num_numas = args.num_numas
    numa_size = args.numa_size
    for cfg in vm_cfg:
        vm = get_vm(cfg.vm_name, si)
        if vm is None:
            print("Virtual Machine with name {} not found".format(cfg.vm_name))
            continue

        if args.print:
            print_pinning(vm.config.extraConfig)
            continue

        num_cpus = vm.summary.config.numCpu
        print(
            "VM {} has {} vCPUs set. Performing pinning".format(cfg.vm_name, num_cpus)
        )

        if not power_off_VM(si, vm, cfg.force):
            print(
                "Denied permission to power off VM {}. Skipping pinning".format(
                    cfg.vm_name
                )
            )
            continue

        extraConfig = remove_old_pinning(vm.config.extraConfig)
        update_config(si, vm, extraConfig)

        if cfg.stride == 0:
            print("New pinning not needed")
            power_on_VM(si, vm)
            continue

        extraConfig = generate_pinned_config(
            extraConfig, cfg.start, num_cpus, cfg.stride, num_numas, numa_size
        )
        update_config(si, vm, extraConfig)

        power_on_VM(si, vm)
