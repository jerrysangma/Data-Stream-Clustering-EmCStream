import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
import time
import umap
import pylab
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, contingency_matrix
from sklearn.cluster import KMeans
import datetime
from statistics import mean

from DataManager import DataManager

import warnings
warnings.filterwarnings('ignore')

verbose=0

randomState1=1
randomState2=1001

ari_threshold = 1.0
ari_threshold_step = 0.001


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    con_mat = contingency_matrix(y_true, y_pred)
    # return purity
    return float(np.sum(np.amax(con_mat, axis=0))) / float(np.sum(con_mat))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print('\nusage: python EmCStream.py datasetFilePath labelsFilePath horizon k\n')
        sys.exit(1)

    print('----- new run at : {} -----'.format(datetime.datetime.now().isoformat()))
    print("Running EmCStream on " + sys.argv[1])
    
    dataManager = DataManager(sys.argv[1], sys.argv[2])
    dataManager.info()
    data_length, feature_cnt = dataManager.data_shape()

    cluster_cnt = int(sys.argv[4])

    ari_drift_threshold = ari_threshold
    initial_ari_drift_threshold = ari_drift_threshold

    consecutiveDriftCnt = 0
    consecutiveNodriftCnt = 0

    increaseDCPCnt = 4 # increase the drift check period when $increaseDCPCnt censuctive NOconcept drifts occur, and increase the ari drift threshold

    decreaseThreshold_DriftCnt = 4 # decrease the ari drift threshold when $decreaseThreshold_DriftCnt censuctive concept drifts occur

    horizon = int(sys.argv[3])
    init_size = horizon*2
    drift_check_period = horizon*5

    initial_drift_check_period = drift_check_period
    
    if horizon > 5 * cluster_cnt:
        match_fnx_size = horizon
    else:
        match_fnx_size = 2*horizon

    detected_dirft_count = 0

    print('init size: {}\nhorizon: {}'.format(init_size, horizon))    #
    print('drift check period : {}'.format(drift_check_period))
    print('overlap size : {}'.format(match_fnx_size))
    print('cluster count : {}'.format(cluster_cnt))
    print('initial ari_drift_threshold : {}'.format(ari_drift_threshold))
    print('\nExecuting EmCStream...\n')
    total_embedding = np.empty(shape=[0, 2])
    total_klabels = [] # kmeans labels
    total_tlabels = [] # true labels
    total_X = np.empty(shape=[0, feature_cnt])

    no_more_data = False
    match_fnx = None
    drift_occured = False
    very_first_loop = True
    in_init_cycle = True
    aris = list()
    purities = list()
    siluets = list()

    max_ari_tobe_threshold = 0

    while not no_more_data:
        # reinit umap when a concept drift occurs
        embedding = None

        in_init_cycle = True
        X, tlabels = dataManager.get_data(init_size)
        if X is False:
            print('breaking the loop because X is false, which means no more data.')
            break
        reducer = umap.UMAP(random_state = randomState1)
        randomState1 = randomState1 + 1
        embedding = reducer.fit_transform(X)
        added_X = X
        added_tlabels = tlabels

        drift_occured = False
        while not drift_occured and not no_more_data:
            # come back here when no concept drift occured
            added_instance_cnt = 0
            drift_check_time = False
            while not drift_check_time and not no_more_data:
                X, tlabels = dataManager.get_data(horizon)
                if X is False:
                    no_more_data = True
                    if verbose:
                        print('setting no_more_data to true.')
                    break
                embedding = np.append(embedding, reducer.transform(X), axis = 0)
                added_X = np.append(added_X, X, axis = 0)
                added_tlabels = np.append(added_tlabels, tlabels) 
                added_instance_cnt += horizon
                if added_instance_cnt >= drift_check_period:
                    drift_check_time = True

            if not no_more_data:
                # it is drift check time
                if verbose:
                    print('calculation kmeans on this window, with k={}'.format(cluster_cnt))
                    print('reinitialize umap with last [{}] data instances, to check for concept drift.'.format(added_instance_cnt))
                kmeans = KMeans(n_clusters=cluster_cnt, random_state=0).fit(embedding)
                klabels = [x for x in kmeans.labels_]
                new_reducer = umap.UMAP(random_state = randomState2)
                randomState2 = randomState2 + 1
                new_embedding = new_reducer.fit_transform(added_X)
                new_kmeans = KMeans(n_clusters=cluster_cnt, random_state=0).fit(new_embedding)
                new_klabels = [x for x in new_kmeans.labels_]

                if not very_first_loop:
                    # create the match fnx
                    prev_tomatch = total_klabels[-match_fnx_size:]
                    curr_tomatch = klabels[:match_fnx_size]
                    match = np.zeros(shape = [cluster_cnt, cluster_cnt])
                    match_fnx = np.zeros(shape = [cluster_cnt])
                    for i in range(match_fnx_size):
                        match[int(curr_tomatch[i]), int(prev_tomatch[i])] += 1
                    for i in range(cluster_cnt):
                        match_fnx[i] = np.argmax(match[i])
                    # match function is ready
                    if verbose:
                        print '---+---+---+---'
                        print match
                        print '---'
                        print match_fnx
                        print '---+---+---+---'
                    # convert labels to match previous labels
                    for i in range(len(klabels)):
                        klabels[i] = match_fnx[klabels[i]]

                ari = adjusted_rand_score(klabels, new_klabels)
                if ari >= ari_drift_threshold:
                    # no drift
                    max_ari_tobe_threshold = 0

                    true_ari = adjusted_rand_score(klabels, added_tlabels)
                    purity = purity_score(klabels, added_tlabels)
                    siluet = silhouette_score(added_X, klabels)
                    aris.append(true_ari)
                    purities.append(purity)
                    siluets.append(siluet)

                    if verbose:
                        print('no concept drift yet [{}]'.format(ari))
                    consecutiveDriftCnt = 0
                    consecutiveNodriftCnt = consecutiveNodriftCnt + 1
                    if consecutiveNodriftCnt >= increaseDCPCnt :
                        consecutiveNodriftCnt = 0
                        if drift_check_period < initial_drift_check_period :
                            drift_check_period = drift_check_period + horizon
                            if verbose:
                                print('new drift check period (increased) is [{}]'.format(drift_check_period))

                    ari_drift_threshold = ari - ari_threshold_step
                    if verbose:
                        print('new ari drift threshold (increased) is [{}]'.format(ari_drift_threshold))

                    if very_first_loop:
                        total_embedding = np.append(total_embedding, embedding, axis = 0)
                        total_klabels = total_klabels + klabels
                        total_tlabels = np.append(total_tlabels, added_tlabels)
                        total_X = np.append(total_X, added_X, axis = 0)

                        very_first_loop = False
                    else:
                        total_embedding = np.append(total_embedding, embedding[match_fnx_size:], axis = 0)
                        total_klabels = total_klabels + klabels[match_fnx_size:]
                        total_tlabels = np.append(total_tlabels, added_tlabels[match_fnx_size:])
                        total_X = np.append(total_X, added_X[match_fnx_size:], axis = 0)

                    if verbose:
                        print('total embedding : [{}]'.format(total_embedding.shape))
                        print('total klabels : [{}]'.format(len(total_klabels)))
                        print('total tlabels : [{}]'.format(total_tlabels.shape))
                        print('total X : [{}]'.format(total_X.shape))
                else:
                    drift_occured = True
                    consecutiveNodriftCnt = 0
                    if in_init_cycle:
                        consecutiveDriftCnt += 1
                        if ari > max_ari_tobe_threshold:
                            max_ari_tobe_threshold = ari
                        if verbose:
                            print('***** a consecutive concept drift detected.*****[{}]'.format(ari))
                        dataManager.iterate_index(-(added_instance_cnt+init_size))
                        if drift_check_period > match_fnx_size + horizon :
                            drift_check_period = drift_check_period - horizon
                            if verbose:
                                print('new drift check period (decreased) is [{}]'.format(drift_check_period))
                        elif consecutiveDriftCnt >= decreaseThreshold_DriftCnt:
                            #ari_drift_threshold = ari_drift_threshold - ari_threshold_step
                            ari_drift_threshold = max_ari_tobe_threshold - ari_threshold_step
                            max_ari_tobe_threshold = 0
                            if verbose:
                                print('new ari drift threshold (decreased) is [{}]'.format(ari_drift_threshold))
                            consecutiveDriftCnt = 0
                    else:
                        if verbose:
                            print('********** a new concept drift detected.**********[{}]'.format(ari))
                        dataManager.iterate_index(-added_instance_cnt)
                        detected_dirft_count += 1

            else:
                # no more data
                if verbose:
                    print('calculating kmeans on this window, with k={}'.format(cluster_cnt))
                kmeans = KMeans(n_clusters=cluster_cnt, random_state=0).fit(embedding)
                klabels = [x for x in kmeans.labels_]

                # create the match fnx
                prev_tomatch = total_klabels[-match_fnx_size:]
                curr_tomatch = klabels[:match_fnx_size]
                match = np.zeros(shape = [cluster_cnt, cluster_cnt])
                match_fnx = np.zeros(shape = [cluster_cnt])
                for i in range(match_fnx_size):
                    match[int(curr_tomatch[i]), int(prev_tomatch[i])] += 1
                for i in range(cluster_cnt):
                    match_fnx[i] = np.argmax(match[i])
                # match function is ready
                if verbose:
                    print '---+---+---+---'
                    print match
                    print '---'
                    print match_fnx
                    print '---+---+---+---'
                # convert labels to match previous labels
                for i in range(len(klabels)):
                    klabels[i] = match_fnx[klabels[i]]

                total_embedding = np.append(total_embedding, embedding[match_fnx_size:], axis = 0)
                total_klabels = total_klabels + klabels[match_fnx_size:]
                total_tlabels = np.append(total_tlabels, added_tlabels[match_fnx_size:])
                total_X = np.append(total_X, added_X[match_fnx_size:], axis = 0)
                break

            embedding = np.empty(shape=[0, 2])
            added_X = np.empty(shape=[0, feature_cnt])
            added_tlabels = np.empty(shape=[0, 1])
            klabels = []
            if not drift_occured:
                dataManager.iterate_index(-match_fnx_size)
            
            in_init_cycle = False

    print('\n\n-\n--\n---\nExecution completed:')
    if verbose:
        print('total embedding : [{}]'.format(total_embedding.shape))
        print('total klabels : [{}]'.format(len(total_klabels)))
        print('total tlabels : [{}]'.format(total_tlabels.shape))
        print('total X : [{}]'.format(total_X.shape))
    ari = adjusted_rand_score(total_klabels, total_tlabels)  
    print('\nari of total calculation is [{}]'.format(ari))   
    print('average ari of total [{}] chunks is [{}]'.format(len(aris), mean(aris)))
    pur = purity_score(total_klabels, total_tlabels)  
    print('\npurity of total calculation is [{}]'.format(pur))   
    print('average purity of total [{}] chunks is [{}]'.format(len(purities), mean(purities)))
    sil = silhouette_score(total_X, total_klabels)  
    print('\nsilhouette_score of total calculation is [{}]'.format(sil))   
    print('average silhouette_score of total [{}] chunks is [{}]'.format(len(siluets), mean(siluets)))
    print('\ndetected drift count is [{}]'.format(detected_dirft_count))

    sys.exit(0)