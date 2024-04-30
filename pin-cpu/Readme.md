# Pinning VMs on VMware platforms
Numactl is primarily used inside operating systems like Linux to pin the workload to physical CPU cores. However, in the case of a virtualized environment, there is another layer of abstraction between the virtual and physical cores. Pinning at this layer will allow similar NUMA strategies to be used as in the case of a non-virtualized environment for optimal performance. 

The following methods can be used to pin a VM:

## VMware vCenter
### Editing the VM properties
Prerequisites: A virtual machine that is powered off.

Select the virtual machine of choice and go into the `: Actions` dropdown menu. Select `Edit Settings`.

Select the `Advanced Parameters` tab on the top.

Add the following attributed for every vCPU to be pinned:
- Attibute: `sched.vcpu_.affinity`. Replace `_` with the vCPU number to be pinned. This will be a value in the range `[0, num_vcpus-1]` where `num_vcpus` is the number of vCPUs allocated to this VM.
- Value: `x`. Replace `x` with the physical thread number to pin it to. The two threads of a hyper-threaded CPU are one after another. So you would want to select every other thread. 

This is an example of a sample pinning for a 112 vCPU VM:
- Attribute: `sched.vcpu0.affinity`, Value: `0`
- Attribute: `sched.vcpu1.affinity`, Value: `2`
- Attribute: `sched.vcpu2.affinity`, Value: `4`
- Attribute: `sched.vcpu3.affinity`, Value: `6`
- ...
- Attribute: `sched.vcpu111.affinity`, Value: `222`

Also add the follwing attribute for making sure ESXi does not try to migrate these threads:
- Attribute: `sched.cpu.affinity.exclusive`
- Value: `TRUE`

Now, power on the VM for the settings to take effect.

To remove the pinning, delete all the attribute and value pairs added as part of pinning.

### Using script
You can use the reconfigure VM script to pin or unpin any VM managed by a VMware vCenter Server. To run the script, `python3` is needed along with the requirements in `requirements.txt`. These can be installed using the command `pip install -r requirements.txt`.

```
$ python reconfigure_vm.py -h

usage: reconfigure_vm.py [-h] [--host HOST] [--user USER] [--password PASSWORD] [--vm-name VM_NAME] [--insecure] [--no-pin] [--no-ht] [-s START] [-t STRIDE] [-f]
                         [-i INPUT] [-n NUM_NUMAS] [-z NUMA_SIZE]

Pin a VM

options:
  -h, --help            show this help message and exit
  --host HOST           vCenter URL. e.g. abc.vcsa.com
  --user USER           Privileged user to reconfigure VM. e.g. administrator@vsphere.local
  --password PASSWORD   Password to the account. Alternatively, use stdin when prompted
  --vm-name VM_NAME     Name of the virtual machine to pin
  --insecure            Ignore TLS certificate verification
  --no-pin              Remove pinning from the VM. Equivalent to stride=0
  --no-ht               Remove pinning from the VM. Equivalent to stride=1
  -s START, --start START
                        Starting CPU/ thread to pin from
  -t STRIDE, --stride STRIDE
                        Option to control size of stride. Default is 2 (HT on)
  -f, --force           Automatically shutdown VM if powered on without prompt
  -i INPUT, --input INPUT
                        CSV file for batch operations
  -n NUM_NUMAS, --num-numas NUM_NUMAS
                        Number of numa nodes to use. If specified, it will pin <cores>/<num_numas> per numa starting from <start> in each numa
  -z NUMA_SIZE, --numa-size NUMA_SIZE
                        Size of numa node. Needed if num-numas is specified
```

The script automatically tries to reboot the VM for the settings to take effect. A user prompt of y/yes is required to go through with the reboot operation.

#### Batch Operations
Batch operation for pinning can be performed by using a csv file as input. Each line of the file must contain the following comma seperated values in order:
1. `VM_NAME`: Name of the VM to pin
1. `START`: Core to start with
1. `STRIDE`: Stride when pinning. (e.g 2 for HT on, for HT off, 0 for no pinning)
1. `FORCE`: If VM should be automatically turned off (True/False)

### Additional considerations
1. Whenever the number of vCPUs for a VM is needing to be adjusted, it has to be unpinned. This can cause a VM to go into invalid state otherwise.
1. There may be some discrepancy in the max vCPUs per VirtualNode. If this is the case, you can manually set this by editing the VM settings. Go into `VM options` tab, select `CPU topology` and edit the cores per socket based on your system. Additionally, you can also set the NUMA nodes manually here.

## Troubleshooting
Specifying wrong configuration for pinning can result in invalid VMs. To resolve this state, do the following steps
1. Unregister the VM from ESXi UI
1. Login to ESXi via SSH and navigate to the directory with the `.vmx` file
1. Remove the pinning settings from the `.vmx` file. This are the lines that mention `sched.vcpu_.affinity` (Replace `_` with every vCPU that is being pinned) and also the line that says `sched.cpu.affinity.exclusive`
1. Register the VM again using the `.vmx` file
