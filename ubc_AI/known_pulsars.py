"""
Aaron Berndsen: a module which reads the ATNF database and
GBNCC discoveries page and provides a list of known pulsars.

Pulsars are given as ephem.FixedBody objects, with all the
ATNF pulsar properties in self.props[key].

"""

import fractions
import os
import re
from urllib.request import urlopen

import numpy as np
from bs4 import BeautifulSoup

topdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(("/").join(topdir.split("/")[:-1]), "data")


def get_allpulsars():
    # BAD
    """
    return dictionary of all pulsars found in
    ATNF, PALFA and GBNCC databases
    """
    allpulsars = {}
    atnf = ATNF_pulsarlist()
    two_join = {
        "gbncc": GBNCC_pulsarlist(),
        "palfa": PALFA_pulsarlist(),
        # "drift": driftscan_pulsarlist(),
        "ao327": ao327_pulsarlist(),
        # "deepmb": deepmb_pulsarlist(),
        "gbt350": GBT350NGP_pulsarlist(),
        "fermi": FERMI_pulsarlist(),
        "lofar": LOFAR_pulsarlist(),
        "ryan": ryan_pulsars(),
    }

    for k, v in atnf.items():
        allpulsars[k] = v
    for survey, lst in two_join.items():
        for k, v in lst.items():
            if k in allpulsars:
                # add it in if P0 differ by 10ms
                try:
                    if int(100 * allpulsars[k].P0) == int(100 * v.P0):
                        print("pulsar %s from %s already in our database" % (k, survey))
                        continue
                    else:
                        k += "a"
                except (ValueError):
                    k += "a"
            allpulsars[k] = v

    return allpulsars


def ATNF_pulsarlist():
    """
    Contact the ATNF pulsar catalogue, returning an array of data. URL may need regular updating.
    Each row corresponding to one pulsar, with columns in the format:

    return:
    dictionary of pulsars, keyed by PSRJ
    """
    # URL to get | NAME | PSRJ | RAJ | DECJ | P0 | DM |
    # OLD: url = "http://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?Name=Name&JName=JName&RaJ=RaJ&DecJ=DecJ&P0=P0&DM=DM&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=&ephemeris=short&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Short+without+errors&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query&table_bottom.x=27&table_bottom.y=10"
    url = "https://www.atnf.csiro.au/people/pulsar/psrcat/proc_form.php?version=1.67&Name=Name&JName=JName&RaJ=RaJ&DecJ=DecJ&P0=P0&DM=DM&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=&ephemeris=short&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Long+with+last+digit+error&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query&table_bottom.x=63&table_bottom.y=22"
    sock = urlopen(url)
    data = sock.read()
    sock.close()
    data = str(data)
    # print(re.findall(r'(\w+[\s\w]*)\b', data))
    # print(data)
    data_dict = {}
    first_count = True
    psr_number = 1
    for datalines in data.split(" "):
        if datalines not in ["", " ", r"<a"] and "nbsp" not in datalines.split("&"):
            if len(datalines.split("\\n")) > 1:
                # If you can turn the split into an int (mostly takes care of the website footer)
                try:
                    # If pulsar number is one (takes care of header info)
                    if int(datalines.split("\\n")[-1]) == 1:
                        first_count = False
                    data_dict[psr_number] = {}
                    count = 0
                except:
                    pass
            else:
                if not first_count:
                    if count == 0:
                        data_dict[psr_number].update(
                            {"Name": str(datalines.split('">')[-1].split("<")[0])}
                        )
                    elif count == 1:
                        data_dict[psr_number].update({"PSRJ": str(datalines)})
                    elif count == 2:
                        data_dict[psr_number].update({"RAJ": str(datalines)})
                    elif count == 4:
                        data_dict[psr_number].update({"DECJ": str(datalines)})
                    elif count == 6:
                        data_dict[psr_number].update({"P0": str(datalines)})
                    elif count == 8:
                        data_dict[psr_number].update({"DM": str(datalines)})
                    count += 1

    pulsars = {}
    for psr_num in data_dict.keys():
        pulsars[data_dict[psr_num]["PSRJ"]] = pulsar(
            *data_dict[psr_num].values(), catalog="ATNF"
        )

    return pulsars


def GBNCC_pulsarlist():
    """
    gather the pulsars listed on the GBNCC discover page

    return:
    dictionary of pulsars keyed by PSRJ
    """
    pulsars = {}
    url = "http://arcc.phys.utb.edu/gbncc/"
    try:
        sock = urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data, "html.parser")
        sock.close()
        rows = soup.findAll("tr")[1:]
    except (IOError):
        rows = []
    for row in rows:
        cols = row.findAll("td")
        name = cols[1].text
        p0 = float(cols[2].text) / 1000.0  # ms --> [s]
        dm = cols[3].text
        coords = name.strip("J")
        if "+" in coords:
            raj = coords.split("+")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("+")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("+%s:00" % match)
                gbncc = True
            else:
                decj = str("+%s:%s" % (match[0:2], match[2:4]))
                gbncc = False
        else:
            raj = coords.split("")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("-")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("-%s:00" % match)
                gbncc = True
            else:
                decj = str("-%s:%s" % (match[0:2], match[2:4]))
                gbncc = False

        pulsars[name] = pulsar(
            name=name,
            psrj=name,
            ra=raj,
            dec=decj,
            P0=p0,
            DM=dm,
            gbncc=gbncc,
            catalog=url,
        )

    return pulsars


def LOFAR_pulsarlist():
    """
    gather the pulsars listed on the LOFAR lotas discovery page
    """
    pulsars = {}
    url = "http://astron.nl/pulsars/lofar/surveys/lotas/"
    try:
        sock = urlopen(url)
        data = sock.read()
        sock.close()
    except (IOError):
        data = ""
    datas = data.splitlines()
    # read until '-----------'
    for n, l in enumerate(datas):
        try:
            if l[0] != "J":
                continue
            ldata = l.split()
            name = ldata[0]
            DM = ldata[1]
            P0 = ldata[2]
            ra = ldata[5]
            dec = ldata[6]
            pulsars[name] = pulsar(
                name=name, psrj=name, ra=ra, dec=dec, P0=P0, DM=DM, catalog="lofar"
            )
        except (IndexError):
            pass

    return pulsars


def PALFA_pulsarlist():
    """
    gather the pulsars listed on the palfa discovery page
    http://www.naic.edu/~palfa/newpulsars/

    return:
    dictionary of pulsars keyed by PSRJ
    """
    pulsars = {}
    url = "http://www.naic.edu/~palfa/newpulsars/"
    try:
        sock = urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data, "html.parser")
        sock.close()
        table = soup.findAll("table")[0]  # pulsars are in first table
        rows = table.findAll("tr")[1:]
    except (IOError):
        rows = []
    pinfo = PALFA_jodrell_extrainfo()
    for row in rows:
        cols = row.findAll("td")
        name = cols[1].text
        #        if name == 'J2010+31': continue #skip single-pulse
        try:
            p0 = float(cols[2].text.strip("~")) / 1000.0  # [ms]-->[s]
        except:
            p0 = np.nan
        if name in pinfo:
            if int(pinfo[name][0] * 1000) == int(p0 * 1000):
                dm = pinfo[name][1]
            else:
                dm = np.nan
        else:
            # the website has updated to include the DM :)
            try:
                dm = float(cols[3].text.strip())
            except:
                dm = np.nan
        coords = name.strip("J")
        if "+" in coords:
            raj = coords.split("+")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("+")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("+%s:00" % match)
                gbncc = True
            else:
                decj = str("+%s:%s" % (match[0:2], match[2:4]))
                gbncc = False
        else:
            raj = coords.split("")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("-")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("-%s:00" % match)
                gbncc = True
            else:
                decj = str("-%s:%s" % (match[0:2], match[2:4]))
                gbncc = False

        pulsars[name] = pulsar(
            name=name,
            psrj=name,
            ra=raj,
            dec=decj,
            P0=p0,
            DM=dm,
            gbncc=gbncc,
            catalog=url,
        )

    return pulsars


def PALFA_jodrell_extrainfo():
    """
    return a dictionary keyed by the pulsar name, with values
    (p0, DM).
    This is because http://www.naic.edu/~palfa/newpulsars/
    doesn't list the DM, but
    http://www.jodrellbank.manchester.ac.uk/research/pulsar/PALFA/
    does.

    """
    pulsars = {}
    url = "http://www.jodrellbank.manchester.ac.uk/research/pulsar/PALFA/"
    try:
        sock = urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data, "html.parser")
        sock.close()
        table = soup.findAll("table")[0]  # pulsars are in first table
        rows = table.findAll("tr")[1:]
    except (IOError):
        rows = []
    for row in rows:
        cols = row.findAll("td")
        name = "J%s" % cols[0].text.strip("_jb.tim")
        dm = float(cols[3].text)
        p0 = float(cols[2].text) / 1000.0  # [ms] --> [s]
        # coords = name.strip("J")
        pulsars[name] = (p0, dm)
    return pulsars


r"""
def driftscan_pulsarlist():
    #I do not think the survey is there anymore, not sure where it went.
    pulsars = {}
    # moved url = 'http://www.as.wvu.edu/~pulsar/GBTdrift350/'
    url = "http://astro.phys.wvu.edu/GBTdrift350/"
    try:
        sock = urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data,"html.parser")
        sock.close()
        table = soup.findAll("table")[0]  # pulsars are in first table
        rows = table.findAll("tr")
    except (IOError, IndexError):
        rows = []

    for row in rows:
        cols = row.findAll("td")
        if len(cols) == 0:
            continue

        name = str(cols[0].text)
        p0 = float(cols[1].text) / 1000.0  # [ms] --> [s]
        dm = float(cols[2].text)
        coords = name.strip("J")
        if "+" in coords:
            raj = coords.split("+")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("+")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("+%s:00" % match)
                gbncc = True
            else:
                decj = str("+%s:%s" % (match[0:2], match[2:4]))
                gbncc = False
        else:
            raj = coords.split("-")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("-")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("-%s:00" % match)
                gbncc = True
            else:
                decj = str("-%s:%s" % (match[0:2], match[2:4]))
                gbncc = False

        pulsars[name] = pulsar(
            name=name,
            psrj=name,
            ra=raj,
            dec=decj,
            P0=p0,
            DM=dm,
            gbncc=gbncc,
            catalog=url,
        )
    return pulsars
"""


def ao327_pulsarlist():
    pulsars = {}
    url = "http://www.naic.edu/~deneva/drift-search/"
    try:
        sock = urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data, "html.parser")
        sock.close()
        table = soup.findAll("table")[0]  # pulsars are in first table
        rows = table.findAll("tr")[1:]
    except (IOError, IndexError):
        rows = []

    for row in rows:
        cols = row.findAll("td")
        name = str(cols[1].text).strip("*")
        try:
            p0 = float(cols[2].text) / 1000.0  # [ms] --> [s]
        except ValueError:
            p0 = np.nan
        dm = float(cols[3].text.strip("~"))
        coords = name.strip("J")
        if "+" in coords:
            raj = coords.split("+")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("+")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("+%s:00" % match)
                gbncc = True
            else:
                decj = str("+%s:%s" % (match[0:2], match[2:4]))
                gbncc = False
        else:
            raj = coords.split("-")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("-")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("-%s:00" % match)
                gbncc = True
            else:
                decj = str("-%s:%s" % (match[0:2], match[2:4]))
                gbncc = False

        pulsars[name] = pulsar(
            name=name,
            psrj=name,
            ra=raj,
            dec=decj,
            P0=p0,
            DM=dm,
            gbncc=gbncc,
            catalog=url,
        )
    return pulsars


r"""
def deepmb_pulsarlist():
    #I do not think the survey is there anymore, not sure where it went.
    pulsars = {}
    url = "http://astro.phys.wvu.edu/dmb/"
    try:
        sock = urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data,"html.parser")
        sock.close()
        table = soup.findAll("table")[0]  # pulsars are in first table
        rows = table.findAll("tr")[1:]
    except (IOError, IndexError):
        rows = []
    for row in rows:
        cols = row.findAll("td")
        name = str(cols[0].text)
        if not name.startswith("J"):
            name = "J%s" % name
        p0 = float(cols[1].text) / 1000.0  # [ms] --> [s]
        dm = float(cols[2].text.strip("~"))
        coords = name.strip("J")
        if "+" in coords:
            raj = coords.split("+")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("+")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("+%s:00" % match)
                gbncc = True
            else:
                decj = str("+%s:%s" % (match[0:2], match[2:4]))
                gbncc = False
        else:
            raj = coords.split("-r")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("-")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("-%s:00" % match)
                gbncc = True
            else:
                decj = str("-%s:%s" % (match[0:2], match[2:4]))
                gbncc = False
        pulsars[name] = pulsar(
            name=name,
            psrj=name,
            ra=raj,
            dec=decj,
            P0=p0,
            DM=dm,
            gbncc=gbncc,
            catalog=url,
        )
    return pulsars
"""


def FERMI_pulsarlist():
    """
    pulsars found in :
    http://arxiv.org/pdf/1205.3089v1.pdf

    """
    url = "http://arxiv.org/pdf/1205.3089v1.pdf"
    lst = [
        "J0023+0923 GBT-350 1FGLJ0023.5+0930 3.05 14.3 0.7 0.14 0.017 BW, ",
        "J0101-6422 Parkes 1FGLJ0101.06423 2.57 12.0 0.6 1.78 0.16 ",
        "J0102+4839 GBT-350 1FGLJ0103.1+4840 2.96 53.5 2.3 1.67 0.18 ",
        "J0307+7443 GBT-350 1FGLJ0308.6+7442 3.16 6.4 0.6 36.98 0.24 ",
        "J0340+4130 GBT-350 1FGLJ0340.4+4130 3.30 49.6 1.8 Isolated ",
        "J0533+67 GBT-820 2FGLJ0533.9+6759 4.39 57.4 2.4 Isolated",
        "J0605+37 GBT-820 2FGLJ0605.3+3758 2.73 21.0 0.7 55.6 0.18",
        "J0614-3329 GBT-820 1FGLJ0614.13328 3.15 37.0 1.9 53.6 0.28 ",
        "J0621+25 GBT-820 2FGLJ0621.2+2508 2.72 83.6 2.3 TBD",
        "J1103-5403 Parkes 1FGLJ1103.95355 3.39 103.9 2.5 Isolated NA [18]",
        "J1124-3653 GBT-350 1FGLJ1124.43654 2.41 44.9 1.7 0.23 0.027 BW, ",
        "J1124-36 GMRT-325 1FGLJ1124.43654 5.55 45.1 1.7 TBD NA",
        "J1142+0119 GBT-820 1FGLJ1142.7+0127 5.07 19.2 0.9 1.58 0.15",
        "J1207-5050 GMRT-610 1FGLJ1207.05055 4.84 50.6 1.5 Isolated?",
        "J1231-1411 GBT-820 1FGLJ1231.11410 3.68 8.1 0.4 1.86 0.19 ",
        "J1301+0833 GBT-820 1FGLJ1301.8+0837 1.84 13.2 0.7 0.27 0.024 BW",
        "J1302-32 GBT-350 1FGLJ1302.33255 3.77 26.2 1.0 0.78 0.15 [16]",
        "J1312+0051 GBT-820 1FGLJ1312.6+0048 4.23 15.3 0.8 38.5 0.18 ",
        "J1514-4946 Parkes 1FGLJ1514.14945 3.59 30.9 0.9 1.92 0.17 ",
        "J1536-4948 GMRT-325 1FGLJ1536.54949 3.08 38.0 1.8 TBD",
        "J1544+4937 GMRT-610 18M3037 2.16 23.2 1.2 0.117 0.018 BW [7]",
        "J1551-06x GBT-350 1FGLJ1549.70659 7.09 21.6 1.0 5.21 0.20 [16]",
        "J1628-3205 GBT-820 1FGLJ1627.83204 3.21 42.1 1.2 0.21 0.16 RB",
        "J1630+37 GBT-820 2FGLJ1630.3+3732 3.32 14.1 0.9 12.5 0.16",
        "J1646-2142 GMRT-325 1FGLJ1645.02155c 5.85 29.8 1.1 23 TBD",
        "J1658-5324 Parkes 1FGLJ1658.85317 2.43 30.8 0.9 Isolated ",
        "J1745+1017 Elsberg 1FGLJ1745.5+1018 2.65 23.9 1.3 0.73 0.014 BW, ",
        "J1747-4036 Parkes 1FGLJ1747.44036 1.64 153.0 3.4 Isolated ",
        "J1810+1744 GBT-350 1FGLJ1810.3+1741 1.66 39.6 2.0 0.15 0.045 BW, ",
        "J1816+4510 GBT-350 1FGLJ1816.7+4509 3.19 38.9 2.4 0.36 0.16 y, RB, ",
        "J1828+0625 GMRT-325 1FGLJ1830.1+0618 3.63 22.4 1.2 6.0 TBD",
        "J1858-2216 GBT-820 1FGLJ1858.12218 2.38 26.6 0.9 46.1 0.22 ",
        "J1902-5105 Parkes 1FGLJ1902.05110 1.74 36.3 1.2 2.01 0.16 ",
        "J1902-70 Parkes 2FGLJ1902.77053 3.60 19.5 0.8 TBD",
        "J2017+0603 Nancay 1FGLJ2017.3+0603 2.90 23.9 1.6 2.2 0.18 ",
        "J2043+1711 Nancay 1FGLJ2043.2+1709 2.38 20.7 1.8 1.48 0.17 ",
        "J2047+1053 GBT-820 1FGLJ2047.6+1055 4.29 34.6 2.0 0.12 0.036 BW, ",
        "J2129-0429 GBT-350 1FGLJ2129.80427 7.62 16.9 0.9 0.64 0.37 RB [16]",
        "J2214+3000 GBT-820 1FGLJ2214.8+3002 3.12 22.5 1.5 0.42 0.014 BW, ",
        "J2215+5135 GBT-350 1FGLJ2216.1+5139 2.61 69.2 3.0 0.17 0.21 RB, ",
        "J2234+0944 Parkes 1FGLJ2234.8+0944 3.63 17.8 1.0 0.42 0.015 BW, ",
        "J2241-5236 Parkes 1FGLJ2241.95236 2.19 11.5 0.5 0.14 0.012 BW, ",
        "J2302+4442 Nancay 1FGLJ2302.8+4443 5.19 13.8 1.2 125.9 0.3 ",
    ]
    ###
    lst2 = [
        "J0023+0923   00:23:30      +09:23:00       0.003050     14.300    MSP   Fermi",
        "J0102+4839   01:02:00      +48:39:00       0.002960     53.500    MSP   Fermi",
        "J0307+7443   03:07:00      +74:43:00       0.003160      6.400    MSP   Fermi",
        "J0340+4130   03:40:24      +41:30:00       0.003300     49.600    MSP   Fermi",
        "J0533+67     05:33:54      +67:59:00       0.004390     57.400    MSP   Fermi",
        "J0605+37     06:05:18      +37:58:00       0.002730     21.000    MSP   Fermi",
        "J0621+25     06:21:12      +25:08:00       0.002720     83.600    MSP   Fermi",
        "J1124-3653   11:24:24      -36:53:00       0.002410     44.900    MSP   Fermi",
        "J1142+0119   11:42:42      +01:19:00       0.005070     19.200    MSP   Fermi",
        "J1207-5050   12:07:00      -50:50:00       0.004840     50.600    MSP   Fermi",
        "J1301+0833   13:01:48      +08:33:00       0.001840     13.200    MSP   Fermi",
        "J1312+0051   13:12:36      +00:51:00       0.004230     15.300    MSP   Fermi",
        "J1514-4946   15:14:06      -49:46:00       0.003590     30.900    MSP   Fermi",
        "J1536-4948   15:36:30      -49:48:00       0.003080     38.000    MSP   Fermi",
        "J1544+4937   15:44:00      +49:37:00       0.002160     23.200    MSP   Fermi",
        "J1551-06     15:51:00      -06:59:00       0.007090     21.600    MSP   Fermi",
        "J1628-3205   16:27:48      -32:05:00       0.003210     42.100    MSP   Fermi",
        "J1630+37     16:30:18      +37:32:00       0.003320     14.100    MSP   Fermi",
        "J1646-2142   16:46:00      -21:42:00       0.005850     29.800    MSP   Fermi",
        "J1658-5324   16:58:48      -53:24:00       0.002430     30.800    MSP   Fermi",
        "J1745+1017   17:45:30      +10:17:00       0.002650     23.900    MSP   Fermi",
        "J1747-4036   17:47:24      -40:36:00       0.001640    153.000    MSP   Fermi",
        "J1810+1744   18:10:18      +17:44:00       0.001660     39.600    MSP   Fermi",
        "J1816+4510   18:16:42      +45:10:00       0.003190     38.900    MSP   Fermi",
        "J1828+0625   18:28:00      +06:25:00       0.003630     22.400    MSP   Fermi",
        "J1858-2216   18:58:06      -22:16:00       0.002380     26.600    MSP   Fermi",
        "J1902-5105   19:02:00      -51:05:00       0.001740     36.300    MSP   Fermi",
        "J1902-70     19:02:42      -70:53:00       0.003600     19.500    MSP   Fermi",
        "J2047+1053   20:47:36      +10:53:00       0.004290     34.600    MSP   Fermi",
        "J2129-0429   21:29:48      -04:29:00       0.007620     16.900    MSP   Fermi",
        "J2215+5135   22:15:00      +51:35:00       0.002610     69.200    MSP   Fermi",
        "J2234+0944   22:34:48      +09:44:00       0.003630     17.800    MSP   Fermi",
        "J1400-56     14:00:00      -56:00:00       0.410700    123.000   YPSR   Fermi",
        "J1203-62     12:03:00      -62:00:00       0.393090    285.000   YPSR   Fermi",
    ]
    pulsars = {}
    for row in lst:
        name = row.split()[0]
        p0 = float(row.split()[3]) / 1000.0  # [ms] --> [s]
        dm = float(row.split()[4])
        coords = name.strip("J")
        if "+" in coords:
            raj = coords.split("+")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("+")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("+%s:00" % match)
                gbncc = True
            else:
                decj = str("+%s:%s" % (match[0:2], match[2:4]))
                gbncc = False
        else:
            raj = coords.split("-")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("-")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("-%s:00" % match)
                gbncc = True
            else:
                decj = str("-%s:%s" % (match[0:2], match[2:4]))
                gbncc = False

        pulsars[name] = pulsar(
            name=name,
            psrj=name,
            ra=raj,
            dec=decj,
            P0=p0,
            DM=dm,
            gbncc=gbncc,
            catalog=url,
        )

    for row in lst2:
        name = row.split()[0]
        p0 = float(row.split()[3])
        dm = float(row.split()[4])
        ra = row.split()[1]
        dec = row.split()[2]
        pulsars[name] = pulsar(
            name=name,
            psrj=name,
            ra=ra,
            dec=dec,
            P0=p0,
            DM=dm,
            gbncc=False,
            catalog="FERMI",
        )
    return pulsars


def GBT350NGP_pulsarlist():
    """
    pulsars found in:
    http://arxiv.org/pdf/0710.1745v1.pdf
    name, P[ms], DM
    """
    url = "http://arxiv.org/pdf/0710.1745v1.pdf"
    lst = [
        "J0033+57 315 76",
        "J0033+61 912 37",
        "J0054+66 1390 15",
        "J0058+6125 637 129",
        "J0240+62 592 4 ",
        "J0243+6027 1473 141",
        "J0341+5711 1888 100",
        "J0408+55A 1837 55 ",
        "J0408+55B 754 64 ",
        "J0413+58 687 57 ",
        "J0419+44 1241 71 ",
        "J0426+4933 922 85",
        "J0519+44 515 52",
        "J2024+48 1262 99",
        "J2029+45 1099 228",
        "J2030+55 579 60",
        "J2038+35 160 58",
        "J2043+7045 588 57",
        "J2102+38 1190 85",
        "J2111+40 4061 120 ",
        "J2138+4911 696 168",
        "J2203+50 745 79",
        "J2208+5500 933 105",
        "J2213+53 751 161",
        "J2217+5733 1057 130",
        "J2222+5602 1336 168",
        "J2238+6021 3070 182",
        "J2244+63 461 92",
        "J2315+58 1061 74",
        "J2316+64 216 248",
        "J2326+6141 790 33",
        "J2343+6221 1799 117",
        "J2352+65 1164 152",
    ]
    pulsars = {}
    for row in lst:
        name, p0, dm = row.split()
        p0 = float(p0) / 1000.0  # [ms] --> [s]
        dm = float(dm)
        coords = name.strip("J")
        if "+" in coords:
            raj = coords.split("+")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("+")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("+%s:00" % match)
                gbncc = True
            else:
                decj = str("+%s:%s" % (match[0:2], match[2:4]))
                gbncc = False
        else:
            raj = coords.split("-")[0]
            raj = str("%s:%s" % (raj[0:2], raj[2:]))
            tmp = coords.split("-")[1]
            match = re.match(r"\d+", tmp).group(0)
            if len(match) == 2:
                decj = str("-%s:00" % match)
                gbncc = True
            else:
                decj = str("-%s:%s" % (match[0:2], match[2:4]))
                gbncc = False
        pulsars[name] = pulsar(
            name=name,
            psrj=name,
            ra=raj,
            dec=decj,
            P0=p0,
            DM=dm,
            gbncc=gbncc,
            catalog=url,
        )
    return pulsars


def ryan_pulsars():
    """
    return the list of pulsars that Ryan Lynch provided

    """
    pulsars = {}
    with open(datadir + "/ryan_lynch_pulsars.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip()
            if "#" in row:
                continue
            elif len(row) < 1:
                continue
            try:
                name, ra, dec, p0, dm, typ, catalog = row.split()
            except (ValueError):
                continue
            if len(dec.split(":")) == 2 or "00:00" in dec:
                # Dec not listed accurately
                gbncc = True
            else:
                gbncc = False
            pulsars[name] = pulsar(
                name=name,
                psrj=name,
                ra=ra,
                dec=dec,
                P0=p0,
                DM=dm,
                gbncc=gbncc,
                catalog=catalog,
            )
    return pulsars


#
#    ______  __ __|  |   ___________ _______
#    \____ \|  |  \  |  /  ___/\__  \\_  __ \
#    |  |_> >  |  /  |__\___ \  / __ \|  | \/
#    |   __/|____/|____/____  >(____  /__|
#    |__|                   \/      \/
#
class pulsar:
    """
    A simple class describing a pulsar.
    Only tracks: name, ra, dec, P0, DM

        ra/dec should be in HH:MM:SS or DD:MM:SS format

    """

    def __init__(self, name, psrj, ra, dec, P0, DM, gbncc=False, catalog=None):
        self.name = name
        self.psrj = psrj
        self.ra = ra
        self.dec = dec
        # keep track if this was listed on GBNCC page
        # since it only gives DEC to nearest degree (used in matching)
        self.gbncc = gbncc
        try:
            self.P0 = float(P0)
        except ValueError:
            self.P0 = np.nan
        try:
            self.DM = float(DM)
        except:
            self.DM = np.nan
        self.catalog = catalog


def hhmm2deg(hhmm):
    """
    given string 'hhmmss' or 'hh:mm:ss' (of varying length), convert to
    degrees

    """
    deg = 0.0
    if ":" in hhmm:
        s = hhmm.split(":")
        ncomp = len(s)
        if ncomp == 1:
            deg = float(s[0]) * 360.0 / 24.0
        elif ncomp == 2:
            deg = (float(s[0]) + float(s[1]) / 60.0) * 360.0 / 24.0
        elif ncomp >= 3:
            deg = (
                (float(s[0]) + float(s[1]) / 60.0 + float(s[2]) / 3600.0) * 360.0 / 24.0
            )
    else:
        if len(hhmm) == 2:
            deg = float(hhmm) * 360.0 / 24.0
        elif len(hhmm) == 4:
            deg = (float(hhmm[0:2]) + float(hhmm[2:4]) / 60.0) * 360.0 / 24.0
        elif len(hhmm) >= 6:
            deg = (
                (float(hhmm[0:2]) + float(hhmm[2:4]) / 60.0 + float(hhmm[4:6]) / 3600.0)
                * 360.0
                / 24.0
            )
    return deg


def ddmm2deg(ddmm):
    """
    given string 'hhmmss' or 'hh:mm:ss', convert to
    degrees

    """
    if "-" in ddmm:
        sgn = -1
    else:
        sgn = 1

    if ":" in ddmm:
        s = ddmm.split(":")
        ncomp = len(s)
        if ncomp == 1:
            deg = abs(float(s[0]))
        elif ncomp == 2:
            deg = abs(float(s[0])) + float(s[1]) / 60.0
        elif ncomp >= 3:
            deg = abs(float(s[0])) + float(s[1]) / 60.0 + float(s[2]) / 3600.0
    else:
        if len(ddmm) == 2:
            deg = abs(float(ddmm))
        elif len(ddmm) == 3:  # matches +?? or -??
            deg = abs(float(ddmm))
        elif len(ddmm) == 4:
            deg = abs(float(ddmm[0:2])) + float(ddmm[2:4]) / 60.0
        elif len(ddmm) >= 6:
            deg = (
                abs(float(ddmm[0:2]))
                + float(ddmm[2:4]) / 60.0
                + float(ddmm[4:6]) / 3600.0
            )
    return deg * sgn


def matches(allpulsars, pulsar, sep=0.6, harm_match=False, DM_match=False):
    """
    given a dictionary of all pulsars and a pulsar object,
    return the objects within 'sep' degrees of the pulsar.

    args:
    allpulsars : dictionary of allpulsars
    pulsar : object of interest
    sep : degrees of separation [default .6 deg]

    Optional:
    harmonic_match : [default False], if True reject match if it is
              not a harmonic ratio
              (we print(the rejected object so you can follow up))
    DM_match : [default False], if True reject match if it is there
               is a 15% difference in DM


    Notes:
    *because GBNCC only gives Dec to nearest degree, we consider
    things close if they are separated by less than a degree in dec,
    and 'sep' in RA
    *if the known pulsar is a 'B' pulsar, we use a sep of max(sep, 5degrees)

    """
    matches = {}
    pra = hhmm2deg(pulsar.ra)
    pdec = ddmm2deg(pulsar.dec)
    orig_sep = sep
    for k, v in allpulsars.items():
        amatch = False

        # find positional matches
        ra = hhmm2deg(v.ra)
        dec = ddmm2deg(v.dec)
        # use very wide "beam" for bright pulsars (the "B" pulsars)
        if v.name.startswith("B"):
            sep = max(orig_sep, 2.5)
        else:
            sep = orig_sep
        dra = abs(ra - pra) * np.cos(pdec * np.pi / 180.0)
        ddec = abs(dec - pdec)
        if v.gbncc or pulsar.gbncc:
            # make sure dec is at least one degree
            if dra <= sep and ddec <= max(1.2, sep):
                amatch = True
        elif dra <= sep and ddec <= sep:
            amatch = True

        # reject nearby objects if they aren't harmonics
        if amatch and harm_match:
            max_denom = 100
            num, den = harm_ratio(
                np.round(pulsar.P0, 5), np.round(v.P0, 5), max_denom=max_denom
            )
            if num == 0:
                num = 1
                den = max_denom
            pdiff = abs(1.0 - float(den) / float(num) * pulsar.P0 / v.P0)
            if pdiff > 1:
                amatch = False
                print("%s != a harmonic match (rejecting)" % k)

        # reject nearby objects if 15% difference in DM
        if amatch and DM_match:
            if (v.DM != np.nan) and (pulsar.DM != 0.0):
                dDM = abs(v.DM - pulsar.DM) / pulsar.DM
                if dDM > 0.26:  # 0.15:
                    amatch = False
                    print("%s has a very different DM (rejecting)" % k)

        # finally, we've passed location, harmonic and DM matching
        if amatch:
            matches[v.name] = v
    return sorted(
        matches.values(), key=lambda x: x.name, reverse=True
    )  # gives a sorted list


def harm_ratio(a, b, max_denom=100):
    """
    given two numbers, find the harmonic ratio

    """
    c = fractions.Fraction(a / b).limit_denominator(max_denominator=max_denom)
    return c.numerator, c.denominator
