from sys import stderr, exit, argv
import numpy as np
from scipy.io import loadmat
import os
from os.path import isfile, isdir, realpath, dirname, exists


def preprocessStringerNeuropixelsData(data_path, output_path):
    # Only accept neurons with at least 40 minutes recording length
    minRecLength = 2400.
    rawDataDir = '{}/neuropixel_data/raw'.format(data_path)

    # rawDataDir = '/data.nst/share/data/stringer_spikes_neuropixels'
    basenameDataFile = 'spks{}_Feb18.mat'
    probeLocationsFileName = 'probeLocations.mat'
    probeBordersFileName = 'probeBorders.mat'
    mouseNumber = 1  # We only analyze Waksman
    numberOfProbes = 8

    # load data
    probeLocations = loadmat('{}/{}'.format(rawDataDir,
                                            probeLocationsFileName))

    probeBorders = loadmat('{}/{}'.format(rawDataDir,
                                          probeBordersFileName), squeeze_me=True)

    # mouseNumber = 1 is Waksman
    mouseNumber = 1
    mouseName = str(
        probeLocations['probeLocations'][0][mouseNumber]['mouseName'][0])

    # print("##### Mouse: {}".format(mouseName))
    spks = loadmat('{}/spks/{}'.format(rawDataDir,
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
        saveDataDir = '{}/neuropixel_data/{}/'.format(output_path, mouseName)
        if not isdir('{}/neuropixel_data'.format(output_path)):
            os.mkdir('{}/neuropixel_data'.format(output_path))
        if not isdir(saveDataDir):
            os.mkdir(saveDataDir)
        if not isdir('{}/spks'.format(saveDataDir)):
            os.mkdir('{}/spks'.format(saveDataDir))

        # all the clusters that were found during spike sorting
        Nneurons = len(wHeights)
        for neuron in range(Nneurons):
            # Spacing of electrodes is 20 mm, but two electrodes have the same height
            ccfIndex = int(wHeights[neuron] / 20 * 2)
            detailedArea = ccfOntology[ccfIndex]
            spiketimes_neuron = np.sort(sptimes[neuron])
            t_start = spiketimes_neuron[0]
            t_end = spiketimes_neuron[-1]
            Trec = t_end - t_start
            rate = len(spiketimes_neuron) / Trec
            if (rate < maxRate and Trec > minRecLength) and rate > minRate:
                validNeuronsAreas[detailedArea] += [[probeNumber, neuron]]
                np.save('{}/spks/spikes-{}-{}.npy'.format(saveDataDir,
                                                          probeNumber, neuron), np.sort(spiketimes_neuron))
    # Save dictionary of valid neurons used for the analysis
    np.save('{}validNeuronsAreas.npy'.format(output_path), validNeuronsAreas)


def preprocessRetinaData(data_path, output_path):
    sampling_rate = 10000.  # 10 kHz
    rawDataDir = '{}/retina_data/raw/mode_paper_data/unique_natural_movie'.format(data_path)
    saveDataDir = '{}/retina_data'.format(output_path)
    if not isdir(saveDataDir):
        os.mkdir(saveDataDir)
    if not isdir('{}/spks'.format(saveDataDir)):
        os.mkdir('{}/spks'.format(saveDataDir))
    data = loadmat('{}/data.mat'.format(rawDataDir))

    # Neuron list
    neurons = data['data'][0][0][2][0][0][2][0]
    N_neurons = neurons[-1]

    # find valid neurons with 0.5Hz < rate < 10 Hz and save their spiketimes
    validNeurons = []
    for neuron in range(N_neurons):
        spiketimes_neuron = data['data'][0][0][2][0][0][1][0][neuron][0] / sampling_rate
        t_start = spiketimes_neuron[0]
        t_end = spiketimes_neuron[-1]
        Trec = t_end - t_start
        rate = spiketimes_neuron.size/Trec
        if rate > minRate and rate < maxRate:
            validNeurons += [neuron]
            np.save('{}/spks/spiketimes_neuron{}.npy'.format(saveDataDir, neuron), spiketimes_neuron)
    np.save('{}/validNeurons.npy'.format(saveDataDir), validNeurons)

    # Start and end times movie (probably)
    # T_0 = data['data'][0][0][3][0][0][0][0][0] / sampling_rate
    # T_f = data['data'][0][0][3][0][0][1][1][0] / sampling_rate
    # T = T_f - T_0

    # Description
    # data['data'][0][0][0]

    # Date
    # data['data'][0][0][1]

    # Sampling rate
    # print(data['data'][0][0][2][0][0][0])

    # Full data
    # data['data'][0][0][2][0][0][1]

    # Short data, but not really sure what this is. Spiketimes are not the same
    # data['data'][0][0][2][0][0][3]


def preprocessCA1Data(data_path, output_path):
    rawDataDir = '{}/CA1_data/raw'.format(data_path)
    saveDataDir = '{}/CA1_data'.format(output_path)
    if not isdir(saveDataDir):
        os.mkdir(saveDataDir)
    if not isdir('{}/spks'.format(saveDataDir)):
        os.mkdir('{}/spks'.format(saveDataDir))
    data = loadmat('{}/ec014.277.spike_ch.mat'.format(rawDataDir))
    sample_rate = 20000.  # 20 kHz sampling rate in seconds
    sptimes = data['sptimes'][0] / sample_rate
    singleunit = data['singleunit'][0]
    end_times = data['t_end'].flatten() / sample_rate
    Nneurons = 85
    validNeurons = []
    for neuron in range(Nneurons):
        if singleunit[neuron] == 1:
            spiketimes_neuron = sptimes[neuron].flatten()
            t_start = spiketimes_neuron[0]
            t_end = end_times[neuron]
            Trec = t_end - t_start
            rate = spiketimes_neuron.size/Trec
            if rate > minRate and rate < maxRate:
                validNeurons += [neuron]
                np.save('{}/spks/spiketimes_neuron{}.npy'.format(saveDataDir, neuron), spiketimes_neuron)
    np.save('{}/validNeurons.npy'.format(saveDataDir), validNeurons)


def preprocessCultureData(data_path, output_path):
    rawDataDir = '{}/culture_data/raw'.format(data_path)
    saveDataDir = '{}/culture_data'.format(output_path)
    if not isdir(saveDataDir):
        os.mkdir(saveDataDir)
    if not isdir('{}/spks'.format(saveDataDir)):
        os.mkdir('{}/spks'.format(saveDataDir))
    spiketimes1 = np.loadtxt("{}/L_Prg035_txt_nounstim.txt".format(rawDataDir))
    spiketimes2 = np.loadtxt("{}/L_Prg036_txt_nounstim.txt".format(rawDataDir))
    spiketimes3 = np.loadtxt("{}/L_Prg037_txt_nounstim.txt".format(rawDataDir))
    spiketimes4 = np.loadtxt("{}/L_Prg038_txt_nounstim.txt".format(rawDataDir))
    spiketimes5 = np.loadtxt("{}/L_Prg039_txt_nounstim.txt".format(rawDataDir))

    spiketimes = np.append(spiketimes1, spiketimes2, axis=0)
    spiketimes = np.append(spiketimes, spiketimes3, axis=0)
    spiketimes = np.append(spiketimes, spiketimes4, axis=0)
    spiketimes = np.append(spiketimes, spiketimes5, axis=0)

    sample_rate = 24.03846169
    times = spiketimes.transpose()[0]
    neurons = spiketimes.transpose()[1]
    # spiketimes in seconds
    times = times/sample_rate/1000
    validNeurons = []
    for neuron in np.arange(1, 61):
        spiketimes_neuron = times[np.where(neurons == neuron)[0]]
        t_start = spiketimes_neuron[0]
        t_end = spiketimes_neuron[-1]
        Trec = t_end - t_start
        rate = spiketimes_neuron.size/Trec
        if (rate < maxRate and rate > minRate):
            validNeurons += [neuron]
            np.save('{}/spks/spiketimes_neuron{}.npy'.format(saveDataDir, neuron), spiketimes_neuron)

    np.save('{}/validNeurons.npy'.format(saveDataDir), validNeurons)
    len(validNeurons)


# During preprocessing, only neurons with an average firing rate between minRate and maxRate (in Hz) are considered for the analysis.
minRate = 0.5
maxRate = 10.

recorded_system = argv[1]
# If data_path not specified, use analysis_data of the repository
if len(argv) > 2:
    data_path = argv[2]
else:
    ESTIMATOR_DIR = '{}/..'.format(dirname(realpath(__file__)))
    data_path = '{}/analysis_data'.format(ESTIMATOR_DIR)
# If output_path not specified, use analysis_data of the repository
if len(argv) > 3:
    output_path = argv[2]
else:
    ESTIMATOR_DIR = '{}/..'.format(dirname(realpath(__file__)))
    output_path = '{}/analysis_data'.format(ESTIMATOR_DIR)

if __name__ == "__main__":
    if recorded_system == 'V1':
        preprocessStringerNeuropixelsData(data_path, output_path)
    if recorded_system == 'Retina':
        preprocessRetinaData(data_path, output_path)
    if recorded_system == 'CA1':
        preprocessCA1Data(data_path, output_path)
    if recorded_system == 'Culture':
        preprocessCultureData(data_path, output_path)
