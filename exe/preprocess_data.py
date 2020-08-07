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

# def preprocessCultureData(data_path, output_path):


data_path = argv[2]
output_path = argv[3]

if __name__ == "__main__":
    if argv[1] == 'V1':
        preprocessStringerNeuropixelsData(data_path, output_path)
    # if argv[1] == 'Retina':
    #     preprocessRetinaData(data_path, output_path)
    # if argv[1] == 'EC':
    #     preprocessECData(data_path, output_path)
    # if argv[1] == 'In_vitro':
    #     preprocessCultureData(data_path, output_path)
    exit(main())
