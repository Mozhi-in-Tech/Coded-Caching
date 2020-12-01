
# importing required libraries
from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi
from collections import defaultdict
from setting import Iter1,Iter2,mu,L_array,Rnw,KperCell,delta,C_E,C_F


##########################################
#generate wirless network with collision model
def generate_network_random(Rnw, delta , KperCell):
    """
    This function generates a random realization of the network with following parameters
    :param Rnw: radiuos of the area
    :param delta: $a_{interf}/a_{cell}$
    :param KperCell: average number of users per cell
    :return:
           K: number of users
           H: number of helpers
           Topology of size HxK: The connection matrix  between helpers and users, solid lines in our paper
           Topology_C of size HxK: The interference links from helpers to users, dashed line in our paper
           user_noninterfer: users index who are not in the interference case
           user_with_collision: users index with interference
           dic_k_h: it is a dictionary   maps user k to connected helpers indices
           dic_h_k: it is a dictionary maps helper $h$ to connected users indices in its coverage area
           dic_k_h_i: this is a dictionary maps use $k$ to interference helpers
           dic_h_k_i: this is a dictionary maps helper $h$ to users who are in the interference zone of that helper
    """
    # generating helpers location
    #   Each cell has  cell_rad and helper is located at center point of its cell.
     
    cell_rad = 200 / np.sqrt(3) # radius of a cell approximately is around 115m
    cell_interf =  cell_rad*(1+delta) # interference radius

    No_real_G=1
    lambdab = 1 / (pi * cell_rad ** 2.0) # density of the helpers
    Mean_No_BS = math.ceil(lambdab * pi * (Rnw ** 2))
    No_BS_vec = np.random.poisson(Mean_No_BS, No_real_G)
    H = No_BS_vec[0]  # number of helper in this realization of the network
    while H == 0:
        No_BS_vec = np.random.poisson(Mean_No_BS, No_real_G)
        H = No_BS_vec[0]
    ab = 2 * pi * np.random.rand(1, H)
    rb = Rnw ** 2 * np.random.rand(1, H)
    xb = np.sqrt(rb) * np.cos(ab)
    yb = np.sqrt(rb) * np.sin(ab)
    xb = xb[0] # location of helpers in x axis
    yb = yb[0] # location of helpers in y axis

    # generating users location

    No_real_G = 1
    lambdab = KperCell / (pi * cell_rad ** 2.0)
    Mean_No_US = math.ceil(lambdab * pi * (Rnw ** 2))
    No_US_vec = np.random.poisson(Mean_No_US, No_real_G)
    K = No_US_vec[0]
    au = 2 * pi * np.random.rand(1, K)
    ru = Rnw ** 2 * np.random.rand(1, K)
    xu = np.sqrt(ru) * np.cos(au)
    yu = np.sqrt(ru) * np.sin(au)
    xu = xu[0] # users location in x axis
    yu = yu[0] # users location in y axis

   # generating topology

    Topology = np.zeros([H, K])
    Topology_C = np.zeros([H, K])


    not_serving_user =[]
    for k in range(K):
        for h in range(H):
            dist = np.sqrt((xb[h] - xu[k]) ** 2 + (yb[h] - yu[k]) ** 2)
            if dist <= cell_rad:
                Topology[h, k] = 1
            if dist <= cell_interf and dist> cell_rad:
                Topology_C[h, k] = 1
        if sum(Topology[:, k]) == 0:
            not_serving_user.append(k)

    # removing users who do not connected to any helper
    Topology = np.delete(Topology, not_serving_user, 1)
    Topology_C = np.delete(Topology_C, not_serving_user, 1)
    xu = np.delete(xu, not_serving_user, 0)
    yu = np.delete(yu, not_serving_user, 0)
    K -= len(not_serving_user)


    user_noninterfer = []
    dic_k_h = defaultdict(list)
    dic_h_k = defaultdict(list)
    dic_k_h_i = defaultdict(list)
    dic_h_k_i = defaultdict(list)
    for k in range(K):
        for h in range(H):
            if Topology[h, k] == 1:
                dic_k_h[k].append(h)
                dic_h_k[h].append(k)
            if Topology_C[h, k] == 1:
                dic_k_h_i[k].append(h)
                dic_h_k_i[h].append(k)
        if len(dic_k_h[k])==1 and len(dic_k_h_i[k])==0:
            user_noninterfer.append(k)
    user_with_collision = list(set(range(K)) - set(user_noninterfer))
    user_with_collision.sort()


    Interference = np.zeros([H, H])

    for k in range(K):
        set1=  set(dic_k_h[k])
        set2 = set(dic_k_h_i[k])
        hk = list(set1.union(set2)  )
        indices_hs = []
        if len(hk)>1:
            for h in hk:
                for hh in hk:
                    if h!=hh:
                        Interference[h,hh]=1
                        Interference[hh,h]=1
    return K, H, Topology, Topology_C, user_noninterfer,user_with_collision,dic_k_h,dic_h_k, dic_k_h_i,dic_h_k_i


######################################################
#code cahcing part
def placement(nG_cache, K):
    """

    :param nG_cache: cache replication factor in our paper denoted by L
    :param K: number of users
    :return:
            user_label_cache: label of each user
            G : the length of   $i$-th cache configuration group for i is in [nG_cache]
            Group_cache:  group of cache configuration   
    """

    user_label_cache = np.random.choice(nG_cache, K)   
    G = np.zeros(nG_cache, dtype=np.int32)
    Group = np.zeros([nG_cache, K], dtype=np.int32)
    Group_cache = []
    for i in range(nG_cache):
        index = np.where(user_label_cache == i)
        G[i] = index[0].shape[0]
        Group_cache.append(index[0])

    return user_label_cache, G, Group_cache


def elia_load(G_h, t, nG_cache):
    """
        This function calculates the number of transmissions in the multi-round scheme.
    :param G_h: number of users in group $i$-th cache configuration
    :param t: coded caching gain
    :param nG_cache: cache replication factor
    :return: Load_T is the number of transmission  of multiround scheme in our paper ( please refer to equation (28) in the paper)
    """
    Load_T = 0
    G_h = np.sort(G_h)[::-1]
    for i in range(1, nG_cache - t + 1):
        Load = comb(nG_cache - i, t, exact=True, repetition=False)
        Load = G_h[i - 1] * Load
        Load_T = Load_T + Load
    return Load_T



#########################################
def avalanche(H,K, t, L, user_nonint, user_with_collision, dic_k_h, dic_h_k, dic_k_h_i, dic_h_k_i ):


    time = 0 # total delivery time in avalanche scheme
    dic_u_call = defaultdict(list)
    for k in range(K):
        dic_u_call[k] = 0
    dic_silent = defaultdict(list) # this dic for helper h returns the number of slot the helper should be silent to avoid collision.
    dic_active = defaultdict(list) # this dic for helper h returns the number of slots the helper is active
    nTrans = np.zeros(H, dtype=np.int32) # number of required transmissions for each helper in current round of its multi rounds
    nRoundsTrans = np.zeros(H, dtype=np.int32) # total number of rounds for each helper

    Serve = np.zeros([H, L], dtype=np.int32) # users who are getting served in the current round
    Buffer = np.zeros([H, L], dtype=np.int32) # users who will scheduled to receive their transmission in next rounds
    Served_user =[] # users who  received their transmissions and they are out of network.
    for h in range(H):
        U_h =  list(set(dic_h_k[h]) -set(user_with_collision))
        Served_user.extend(U_h)
        K_h = len(U_h)
        user_label_cache_sorted_h, G_h, Group_cache_h = placement(L, K_h) # cache configuratio of users in K_h
        G_h_next = [max(G_h[it] - 1, 0) for it in range(L)]
        Serve[h, :] = G_h - G_h_next # maximum L users from K_h will get served in the current round and others will stay in Buffer
        nTrans[h] = elia_load(Serve[h, :], t, L)
        Buffer[h, :] = G_h_next # remaining user of K_h\Serve will remain in the buffer and will be scheduled in next rounds
        nRoundsTrans[h] = elia_load(Buffer[h, :], t, L) + nTrans[h]
        if(nRoundsTrans[h]>0): # if helper delivery list in not empty -> add helper to active set
            dic_active[h].extend(list(range(time,time+nTrans[h])))

    user_label, G_h, Group_cache_h = placement(L, K)
    user_to_add = {}

    # find the time slot there is a change in active helpers state
    h_update = np.array([])
    while len(Served_user) != K: #if there is a user which is not served yet continue otherwise terminate the avalacne scheme and return time
        Served_user.sort()

        if (nTrans > 1).all(): #  moving one slot by slot to see when at least a helper finishes its scheduled transmissions in current round.
            val_min = np.min(nTrans) - 1
            index = np.where(nTrans == val_min)
            hs_min = index[0]
            time = time + val_min
            nTrans = np.array([max(nTrans[h] - val_min, 0) for h in range(H)])
            for h in range(H):
                while dic_active[h] and dic_active[h][0] < time:
                    dic_active[h].pop(0)

                while dic_silent[h] and dic_silent[h][0] < time:
                    dic_silent[h].pop(0)

        # moving one time slot
        for h in range(H):
            if dic_active[h]:
                dic_active[h].pop(0)

            if time in dic_silent[h]:
                dic_silent[h].pop(0)
        nTrans = np.array([max(nTrans[h] - 1, 0) for h in range(H)])
        time = time + 1

        # find helpers who finish their transmissions in current round!
        index = np.where(nTrans == 0)
        h_update = index[0]
        h_update = np.random.permutation(h_update)
        
        for h in h_update: # if there are some helpers who finished their transmissions in curent round
            if not dic_silent[h]: # if we did not force  helper h to be silent

                U_C_h = list(set(dic_h_k[h]) - set(user_nonint)) # users who are not served yet and they have connection with helper h
                non_served_users = [user for user in U_C_h if user not in Served_user]
                U_C_h = np.random.permutation(non_served_users)
                if U_C_h.shape[0] != 0:
                    for kc in U_C_h:

                        # here we are checking whether this interference user kc can be added to delivery list of its connected helper
                        # without interfering
                        flag_k= is_possible_to_serve(h, kc, dic_k_h[kc], dic_k_h_i[kc], dic_active, dic_silent, nTrans, Buffer, user_label[kc] )
                        if flag_k == 1 and (user_label[kc] not in user_to_add):
                            user_to_add[user_label[kc]] = kc   # this is a set of new users we need to add to delivery list of helpers

                for it in range(L):
                    if it in user_to_add:
                        Buffer[h][it] = 1
                        kc = user_to_add[it]
                        Served_user.append(kc)

                # adding user_to_add to delivey list
                # and also if buffer of helper is not empty we are loading users form buffer to serve in next round

                G_h_next = [max(Buffer[h][it] - 1, 0) for it in range(L)]
                Serve[h, :] = Buffer[h] - G_h_next
                nTrans[h] = elia_load(Serve[h, :], t, L)
                Buffer[h, :] = G_h_next
                dic_active[h].extend(list(range(time, time + nTrans[h])))
                dic_active[h]=list(set(dic_active[h]))
                dic_active[h].sort()

                # here we are making sure the helper who are interfering with new users list user_to_add will stay silent until end of their transmssions
                # dic_silent : this dictionary for helper h retunrs how many slots that helper  should stay silent
                for it in range(L):
                    if it in user_to_add:
                        kc = user_to_add[it]
                        del(user_to_add[it])
                        all_relay = dic_k_h[kc] + dic_k_h_i[kc]
                        all_others = list(set(all_relay) - set([h]))
                        for hc in all_others:
                            dic_silent[hc].extend(list(range(time, time + nTrans[h])))
                            dic_silent[hc] = list(set(dic_silent[hc]))
                            dic_silent[hc].sort()


    time = time + np.max(nTrans)

    return time

def  is_possible_to_serve(h, k,  k_h, k_h_i, dic_active, dic_silent,nTrans, Buffer , label_k):
    """
      This function checks whether is it possible to add user k to the delivery list of helper h or not
    there are two conditions :
    1) buffer of connected helper for label_k should be empty
    2) helpers who were causing interference to this user should not have any transmission and their buffer of users should be empty
    :param h: connected helper to user k
    :param k: new interference user
    :param k_h: connected helper to user k
    :param k_h_i: helpers who are causing interference to user k
    :param dic_active:
    :param dic_silent: 
    :param nTrans: number of transmissions of all active helpers in its current round
    :param Buffer: Buffer of all helpers
    :param label_k: cache replication group of user k
    :param time:
    :return: True if it is possible otherwise false
    """
    flag = 0 #true

    # checking first condition
    if Buffer[h][label_k] ==0:
        flag = 1

    # checkign second condition
    if flag==1:
        all_relay = list(k_h) + list(k_h_i)
        all_others = list(set(all_relay)-set([h]))

        if all_others:
            for hc in all_others:
                if len(dic_active[hc])>0 or np.sum(Buffer[hc])>0:
                    flag =0


    return flag
#########################################
# montocarlo simulation



output1 = np.zeros([Iter1 * Iter2, len(L_array)])



for it1 in range(Iter1):

    K, H, Topology, Topology_C, user_noninterfer, user_with_collision, dic_k_h, dic_h_k, dic_k_h_i, dic_h_k_i = generate_network_random(Rnw, delta, KperCell)


    for it2 in range(Iter2):
        for ell in range(len(L_array)):
            L = L_array[ell]
            t = L*mu
            user_label, P_norm, P = placement(L, K)
            if t.is_integer():
                t = int(t)
                Fsize = 1 / comb(L, t, exact=True, repetition=False) # normalized by file size

                time_1 = avalanche(H,K, t, L, user_noninterfer, user_with_collision, dic_k_h, dic_h_k, dic_k_h_i, dic_h_k_i )
                output1[it1 * Iter2 + it2, ell] = time_1*Fsize * max(1/C_E, 1/C_F)



            else:


                t1 = math.floor(t)
                t2 = t1 + 1
                alpha = (t2 - L * mu) / (t2 - t1)
                Fsize1 = 1 / comb(L, t1, exact=True, repetition=False)
                time1 = avalanche(H, K, t1, L, user_noninterfer, user_with_collision, dic_k_h, dic_h_k, dic_k_h_i,dic_h_k_i)
                time1 = time1 * Fsize1 * max(1 / C_E, 1 / C_F)
                Fsize2 = 1 / comb(L, t2, exact=True, repetition=False)
                time2 = avalanche(H, K, t2, L, user_noninterfer, user_with_collision, dic_k_h, dic_h_k, dic_k_h_i,dic_h_k_i)
                time2 = time2 * Fsize2 * max(1 / C_E, 1 / C_F)
                time= alpha*time1+(1-alpha)*time2
                output1[it1 * Iter2 + it2, ell] = time
out1 = np.mean(output1, axis =0)

                
########################################################################



print("outputs")

print("L_array (x axis)",L_array)
print("Avalanch delivery time (y axis)",out1)







