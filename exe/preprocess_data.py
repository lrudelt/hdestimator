from sys import stderr, exit, argv
import numpy as np
from scipy.io import loadmat

# During preprocessing, only neurons with an average firing rate between minRate and maxRate (in Hz) are considered for the analysis.
minRate = 0.5
maxRate = 10.


def preprocessStringerNeuropixelsData(data_path, output_path):
    minRecLength = 2400.
    rootDataDir = data_path
    # rootDataDir = '/data.nst/share/data/stringer_spikes_neuropixels'
    spikeDataDir = 'spks'
    basenameDataFile = 'spks{}_Feb18.mat'
    probeLocationsFileName = 'probeLocations.mat'
    probeBordersFileName = 'probeBorders.mat'
    mouseNumber = 1  # We only analyze Waksman
    numberOfProbes = 8

    # load data
    probeLocations = loadmat('{}/{}'.format(rootDataDir,
                                            probeLocationsFileName))

    probeBorders = loadmat('{}/{}'.format(rootDataDir,
                                          probeBordersFileName), squeeze_me=True)

    # for mouseNumber in range(numberOfMice):
    mouseNumber = 1
    mouseName = str(
        probeLocations['probeLocations'][0][mouseNumber]['mouseName'][0])

    # print("##### Mouse: {}".format(mouseName))
    spks = loadmat('{}/spks/{}'.format(rootDataDir,
                                       basenameDataFile.format(mouseName)), squeeze_me=True)
    # find detailed areas from which was recorded in the respective mouse (Waksman)
    detailedAreas = np.array([])
    for probeNumber in range(numberOfProbes):
        ccfOntology = [name[0][0] for name in probeLocations['probeLocations']
                       [0][mouseNumber]['probe'][0][probeNumber]['ccfOntology']]
        detailedAreas = np.append(detailedAreas, np.unique(ccfOntology))
    detailedAreas = np.unique(detailedAreas)
    validNeuronsAreas = {}
    for detailedArea in detailedAreas:
        validNeuronsAreas[detailedArea] = []

    for probeNumber in range(numberOfProbes):
        print("### Probe {}".format(probeNumber))
        ccfCoords = probeLocations['probeLocations'][0][mouseNumber]['probe'][0][probeNumber]['ccfCoords']
        ccfOntology = [name[0][0] for name in probeLocations['probeLocations']
                       [0][mouseNumber]['probe'][0][probeNumber]['ccfOntology']]
        unsortedSptimes = spks['spks'][probeNumber][0]
        clusterIdentities = np.array(
            spks['spks'][probeNumber][1]) - 1  # start at 0 instead of 1
        # cluster heights in microns
        wHeights = spks['spks'][probeNumber][2]
        # Load spikes
        sptimes = [[] for cli in np.unique(clusterIdentities)]
        for sptime, cli in zip(unsortedSptimes, clusterIdentities):
            sptimes[cli] += [float(sptime)]
        # save valid neurons with allowed rates in the dictionary with the right area'
        saveDataDir = '{}/{}/'.format(output_path, mouseName)
        # saveDataDir = '/data.nst/lucas/history_dependence/Data_Neuropixel/{}/spks/'.format(
        #     mouseName)
        for i in range(len(wHeights)):
            ccfIndex = wHeights[i] / 20 * 2
            detailedArea = ccfOntology[ccfIndex]
            spiketimes = np.sort(sptimes[i])
            Trec = spiketimes[-1] - spiketimes[0]
            rate = len(spiketimes) / Trec
            if (rate < maxRate and Trec > minRecLength) and rate > minRate:
                validNeuronsAreas[detailedArea] += [[probeNumber, i]]
                np.save('{}/spks/spikes-{}-{}.npy'.format(saveDataDir,
                                                          probeNumber, i), np.sort(spiketimes))
    # Save dictionary of valid neurons used for the analysis
    np.save('{}validNeuronsAreas.npy'.format(saveDataDir), validNeuronsAreas)


# def preprocessRetinaData(data_path, output_path):

#
# def preprocessECData(data_path, output_path):
#     # Here, only compute the valid neurons, spikes can by directly obtained from the matlab file

# data_path = "/data.nst/lucas/history_dependence/paper/EC_data/spks"
# data = loadmat('%s/ec014.277.spike_ch.mat' % data_path)
# sample_rate = 20000.  # in seconds
# sptimes = data['sptimes'][0] / sample_rate
# singleunit = data['singleunit'][0]
# t_f = data['t_end'] / sample_rate
# t_i = []
# rate = []
#
# for i in range(85):
#     if sptimes[i].size > 10:
#         rate += [sptimes[i].size]
#         t_i += [sptimes[i][9]]
#     else:
#         rate += [sptimes[i].size]
#         t_i += [sptimes[i][0]]
#
# t_i = np.array(t_i).flatten()
# T = t_f - t_i
# rate = np.array(rate) / T
#

# valid_neurons = []
# for i in range(85):
#     if singleunit[i] == 1:
#         print rate[0][i]

# valid_neurons = []
# for i in range(85):
#     if singleunit[i] == 1:
#         if rate[0][i] > 0.5:
#             if rate[0][i] < 10:
#                 valid_neurons += [i]
#

# Choose neuron 61 (rate 2.2 Hz) and neuron 45 (rate 1.8 Hz)

# sptimes_multi = []
# sptimes_single = []
# neurons_multi = []
# neurons_single = []
# for i in range(85):
#     if singleunit[i] == 0:
#         sptimes_multi = np.append(sptimes_multi, sptimes[i])
#         neurons_multi = np.append(neurons_multi, np.zeros(sptimes[i].size) + i)
#     else:
#         if len(sptimes[i]) / (sptimes[i][-1] - sptimes[i][0]) < 10:
#             sptimes_single = np.append(sptimes_single, sptimes[i])
#             neurons_single = np.append(
#                 neurons_single, np.zeros(sptimes[i].size) + i)
# neurons_single = neurons_single + 1
# neurons_multi = neurons_multi + 1
#
# analyzed_neuron = 46  # index of neuron that is analyzed
# spiketimes_analyzed = sptimes[analyzed_neuron - 1].flatten()

# def preprocessCultureData(data_path, output_path):

recorded_system = argv[1]
data_path = argv[2]
output_path = argv[3]


if __name__ == "__main__":
    if recorded_system == 'V1':
        preprocessStringerNeuropixelsData(data_path, output_path)
    # if recorded_system == 'Retina':
    #     preprocessRetinaData(data_path, output_path)
    # if recorded_system == 'EC':
    #     preprocessECData(data_path, output_path)
    # if recorded_system == 'In_vitro':
    #     preprocessCultureData(data_path, output_path)
    exit(main())
