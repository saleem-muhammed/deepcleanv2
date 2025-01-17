#!/bin/python3

import glob
from gwpy.timeseries import TimeSeriesDict
from lalframe import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ifo", default=None, help="The Name of the detector.")
parser.add_argument("-hs", "--hoft_source", default=None, help="The directory of the source gwf files of the strain channel.")
parser.add_argument("-ds", "--detchar_source", default=None, help="The directory of the source gwf files of the strain channel.")
parser.add_argument("-hd", "--hoft_destination", default=None, help="The directory to output the 1-second gwf files of the strain channel.")
parser.add_argument("-dd", "--detchar_destination", default=None, help="The directory to output the 1-second gwf files of the witness channels.")
parser.add_argument("-s", "--start", default=None, type=int, help="The start GPS time.")
parser.add_argument("-D", "--duration", default=None, type=int, help="The duration of the replay data.")
parser.add_argument("-k", "--kind", default=None, type=str, help="The kind of the output 1-second gwf files, either llhoft or lldetchar.")
args = parser.parse_args()

def make_ll_gwf(
        ifo: str,
        source: str,
        channels: list,
        start: int,
        end: int,
        destination: str,
        kind:str
    ):
    data = TimeSeriesDict.read(
        source=source,
        channels=channels,
        start=start,
        end=end,
    )
    
    print(f"Making 1-second gwf files of the following channels:/n")
    print(channels)
    duration = end - start
    i = 0
    while i < duration:
        ll = data.copy()
        lldata = ll.crop(start + i, start + (i+1), copy=False)
        lldata.write(f'{destination}/{ifo[0]}-{ifo}_{kind}-{int(lldata[channels[0]].t0.value)}-{int(lldata[channels[0]].duration.value)}.gwf', format='gwf')
        i += 1

    print("Done!")

def main():
    ifo = args.ifo
    hoft_source = args.hoft_source
    detchar_source = args.detchar_source
    start = int(args.start)
    duration = int(args.duration)
    hoft_destination = args.hoft_destination
    detchar_destination = args.detchar_destination
    kind = args.kind

    hoft_list = glob.glob(f'{hoft_source}/*.gwf')
    hoft_list = sorted(hoft_list)
    detchar_list = glob.glob(f'{detchar_source}/*.gwf')
    detchar_list = sorted(detchar_list)

    strain_ch = [f"{ifo}:GDS-CALIB_STRAIN"]
    wit_chs = utils.frtools.get_channels(detchar_list[0])

    end = start + duration
    if kind == "llhoft":
        source = hoft_list
        channels = strain_ch
        destination = hoft_destination
    if kind == "lldetchar":
        source = detchar_list
        channels = wit_chs
        destination = detchar_destination

    make_ll_gwf(
        ifo=ifo,
        source=source,
        channels=channels,
        start=start,
        end=end,
        destination=destination,
        kind=kind,
    )
    # if duration <= 4096:
    #     make_ll_gwf(
    #         ifo,
    #         f_list,
    #         ch_list,
    #         start,
    #         end,
    #         destination,
    #         tag
    #     )
    # else:
    #     st_list = [st for st in range(start, end, 4096)]    
    #     ed_list = [st + 4096 for st in st_list]
    #     ed_list[-1] = end
    #     for st, ed in zip(st_list, ed_list):
    #         make_ll_gwf(
    #             ifo,
    #             f_list,
    #             ch_list,
    #             st,
    #             ed,
    #             destination,
    #             tag
    #         )


if __name__ == "__main__":
    main()
