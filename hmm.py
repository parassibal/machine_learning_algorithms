
from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        alpha[:,0]=self.pi*self.B[:,self.find_item(Osequence)[0]]
        for t in range(1,L):
            for s in range(S):
                alpha_a=np.dot(self.A[:,s],alpha[:,t-1])
                alpha[s,t]=self.B[s,O[t]]*alpha_a

        return(alpha)

    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        beta[:,len(Osequence)-1]=np.ones(S)
        for t in range(len(Osequence)-2,-1,-1):
            for s in range(S):
                beta_b=self.B[:,self.find_item(Osequence)[t+1]]*beta[:,t+1]
                beta_a_b=self.A[s,:]*beta_b
                beta[s,t]=np.sum(beta_a_b)
        return(beta)


    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        alpha=self.forward(Osequence)
        beta=self.backward(Osequence)
        prob=0
        for i in range(len(self.pi)):
            prob=prob+(alpha[i,len(Osequence)-1]*beta[i,len(Osequence)-1])
        return(prob)


    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        alpha=self.forward(Osequence)
        beta=self.backward(Osequence)
        return((alpha*beta)/self.sequence_prob(Osequence))


    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        alpha=self.forward(Osequence)
        beta=self.backward(Osequence)
        for i in range(S):
            for j in range(S):
                for t in range(L-1):
                    k=self.obs_dict[Osequence[t+1]]
                    prob[i][j][t]=alpha[i][t]*self.A[i][j]*self.B[j][k]*beta[j][t+1]
        prob=prob/self.sequence_prob(Osequence)
        return(prob)


    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        len_pi=len(self.pi)
        Len_o=len(Osequence)
        del_val=np.zeros([len_pi,Len_o])
        for i in range(len(self.pi)):
            del_val[i,0]=self.pi[i]*self.B[i,self.find_item(Osequence)[0]]
        del_val1=np.zeros([len_pi,Len_o-1])
        new_dict={}
        z_val=[]
        item_o=self.find_item(Osequence)
        for key,val in self.state_dict.items():
            new_dict[val]=key
        for t in range(1,len(Osequence)):
            for i in range(len(self.pi)):
                temp=self.A[:,i]*del_val[:,t-1]
                del_val1[i,t-1]=np.argmax(temp)
                del_val[i,t]=self.B[i,item_o[t]]*np.max(temp)
        temp=np.argmax(del_val[:,len(Osequence)-1])
        z_val.append(temp)
        #traverse in back direction
        for t in range(len(Osequence)-2,-1,-1):
            z_val.append(int(del_val1[z_val[-1],t]))
        path=[]
        for i in reversed(z_val):
            path.append(new_dict[i])
        return(path)



    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
