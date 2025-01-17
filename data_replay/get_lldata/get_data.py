#!/bin/python

from gwpy.timeseries import TimeSeries
from gwpy.timeseries import TimeSeriesDict
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ifo", default=None, help="The name of the detector.")
parser.add_argument("-c", "--channels", default=None, help="The text file that records the channels to be fetched.")
parser.add_argument("-s", "--start", default=None, type=int, help="The start GPS time.")
parser.add_argument("-du", "--duration", default=None, type=int, help="The duration of the data.")
parser.add_argument("-l", "--length", default=None, type=int, help="The length of the each gwf files written in the destination.")
parser.add_argument("-d", "--destination", default=None, help="The directory to output the gwf files of the fetched data.")
parser.add_argument("-ht", "--hoft_tag", default=None, help="The tag for the fetched strain channel.")
parser.add_argument("-dt", "--detchar_tag", default=None, help="The tag for the fetched witness channels.")
parser.add_argument("-k", "--kind", default=None, type=str, help="The kind of the output 1-second gwf files, either llhoft or lldetchar.")
args = parser.parse_args()

def fetch(
        ifo: str,
        channels: list,
        start: int,
        end: int,
        destination: str,
        tag: str,
        kind: str,
):
    ts_dict = TimeSeriesDict.get(
        channels=channels,
        start=start,
        end=end,
        nproc=8,
        allow_tape=True,
    )
    ts_dict.write(f"{destination}/{ifo}_{tag}/{ifo[0]}-{ifo}_{kind}-{start}-{end-start}.gwf")

def main():
    ifo = args.ifo
    channels = args.channels
    start = args.start
    duration = args.duration
    end = start + duration
    length = args.length
    destination = args.destination
    hoft_tag = args.hoft_tag
    detchar_tag = args.detchar_tag
    kind = args.kind

    with open(channels, 'r') as f:
        ch_config = f.read()

    ifo_find = re.compile(ifo+':'+r'.*')
    ifo_chlist = ifo_find.findall(ch_config)

    if kind == "llhoft":
        channels = ifo_chlist[:2]
        tag = hoft_tag
    if kind == "lldetchar":
        channels = ifo_chlist[2:]
        tag = detchar_tag
    print("The following channels will be fetched:\n")
    print(channels)

    for t0 in range(start, end, length):
        st = t0
        if end - st < length:
            ed = end
        else:
            ed = t0 + length

        fetch(
            ifo=ifo,
            channels=channels,
            start=st,
            end=ed,
            destination=destination,
            tag=tag,
            kind=kind,
        )

        print("Done!")

if __name__ == "__main__":
    main()
