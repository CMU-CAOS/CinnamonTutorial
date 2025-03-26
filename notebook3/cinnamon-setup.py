import sst
import sys
import getopt
import argparse


################################
# Default values of Configuration
################################
memBandwidth = 2 # in TB/s
memLatency = 128 # in ns
linkBandwidth = 0.25 # in TB/s
linkLatency = 128# in ns
vectorRegs = 256 # number of registers
chips = 4
instructions_dir = ""
################################


def main():
    # Define SST core options
    sst.setProgramOption("timebase", "1ps")
    sst.setProgramOption("stop-at", "5s")

    CHIPS = []

    # Set the Number of Functional Units in the Cinnamon Chips
    numAddUnits = 2
    numMulUnits = 2
    numNttUnits = 1
    numTraUnits = 1
    numBcuUnits = 1
    numBcuBuffs = 1
    numEvgUnits = 2
    

    # The ring dimension of the polynomials
    ringDimension = 64*1024
    # The vector width of the processor
    vectorWidth = 1024
    vectorDepth = ringDimension//vectorWidth

    # The maximum number of hops needed to go from one chip to another
    if chips == 1:
        hops = 0
    elif chips == 4:
        hops = 2
    elif chips == 8:
        hops = 4
    elif chips == 12:
        hops = 2 # 2 hops because Cinnamon-12 uses a switch
    else:
        raise Exception("Unsupported Number of Chips")

    # Define the simulation components
    accelerator = sst.Component("accelerator", "cinnamon.Accelerator")
    accelerator.addParams({
        "clock": "1GHz",
        "num_chips": str(chips),
        "verbose": "1",
        "vec_depth": f"{vectorDepth}",
    })

    network = accelerator.setSubComponent("network","cinnamon.Network")
    network.addParams({
        "verbose": "1",
        "linkBW" : f"{linkBandwidth}TB/s",
        "hops" : f"{hops}",
    })

    for i in range(chips):
        ch = accelerator.setSubComponent(f"chip_{i}","cinnamon.Chip")
        ch.addParams({
            "verbose": "1",
            "numVectorRegs": f"{vectorRegs}",
            "numScalarRegs": "64",
            "memoryRequestWidth": f"2048",
            "numTraUnits" : f"{numTraUnits}",
            "numMulUnits" : f"{numMulUnits}",
            "numNttUnits" : f"{numNttUnits}",
            "numBcuUnits" : f"{numBcuUnits}",
            "numBcuBuffs" : f"{numBcuBuffs}",
            "numAddUnits" : f"{numAddUnits}",
            "numEvgUnits" : f"{numEvgUnits}",
        })
        CHIPS.append(ch)
        reader = ch.setSubComponent("reader","cinnamon.CinnamonTextTraceReader")
        reader.addParams({
            "file": f"{instructions_dir}/instructions{i}"
        })

        iface = ch.setSubComponent("memory", "memHierarchy.standardInterface")

 
        comp_memctrl = sst.Component(f"memory_{i}", "memHierarchy.MemController")
        comp_memctrl.addParams({
            "clock": f"{memBandwidth/2}GHz",
            "addr_range_start": 0,
            "backing": "none",
            "backendConvertor.request_width": f"2048",
        })

        memory = comp_memctrl.setSubComponent("backend", "memHierarchy.simpleMem")
        memory.addParams({
            "access_time": f"{memLatency-2}ns",
            "mem_size":  "4TiB",
            "request_width": f"2048" 
        })


        link_mem_bus_link = sst.Link(f"link_mem_bus_link_{i}")
        link_mem_bus_link.connect( (iface, "port", "1000ps"), (comp_memctrl, "direct_link", "1000ps") )

    for i in range(chips):
        link_chip_network_link = sst.Link(f"chip_network_link_{i}")
        network.addLink(link_chip_network_link,f"chip_port_{i}",f"{linkLatency}ns")
        CHIPS[i].addLink(link_chip_network_link,"cinnamon_network_port",f"{linkLatency}ns")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vregs",type=int,default=256)
    parser.add_argument("--chips",type=int,default=1)
    parser.add_argument("--linkBW",type=float,default=0.25)
    parser.add_argument("--instructions_dir",type=str)
    args = parser.parse_args()
    vregs = args.vregs
    chips = args.chips
    linkBandwidth = args.linkBW
    instructions_dir = args.instructions_dir
    print(f"VectorRegs: {vectorRegs}")
    print(f"Chips: {chips}")
    print(f"LinkBandwidth: {linkBandwidth}TB/s")
    print(f"MemBandwidth: {memBandwidth}TB/s")
    main()
