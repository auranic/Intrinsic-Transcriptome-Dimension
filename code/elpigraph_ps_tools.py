import igraph
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score
from numpy.lib.stride_tricks import as_strided
import math
import elpigraph

def convert_elpigraph_to_igraph(elpigraph):
    edges = elpigraph['Edges'][0]
    nodes_positions = elpigraph['NodePositions']
    g = igraph.Graph()
    g.add_vertices(len(nodes_positions))
    g.add_edges(edges)
    return g

def extract_trajectories(tree,root_node,verbose=False):
    edges = tree['Edges'][0]
    nodes_positions = tree['NodePositions']
    g = igraph.Graph()
    g.add_vertices(len(nodes_positions))
    g.add_edges(edges)
    degs = g.degree()
    leaf_nodes = [i for i,d in enumerate(degs) if d==1]
    if verbose:
        print(len(leaf_nodes),'trajectories found')
    all_trajectories = []
    for lf in leaf_nodes:
        path_vertices=g.get_shortest_paths(root_node,to=lf,output='vpath')
        all_trajectories.append(path_vertices[0])
        path_edges=g.get_shortest_paths(root_node,to=lf,output='epath')
        if verbose:
            print('Vertices:',path_vertices)
            print('Edges:',path_edges)
        ped = []
        for ei in path_edges[0]:
            ped.append((g.get_edgelist()[ei][0],g.get_edgelist()[ei][1]))
        if verbose:
            print('Edges:',ped)
        # compute pseudotime along each path
    return all_trajectories

def pseudo_time(root_node,point_index,traj,projval,edgeid,edges):
    xi = int(point_index)
    proj_val_x = projval[xi]
    #print(proj_val_x)
    if proj_val_x<0:
        proj_val_x=0
    if proj_val_x>1:
        proj_val_x=1
    edgeid_x = edgeid[xi]
    #print(edges[edgeid_x])
    traja = np.array(traj)
    i1 = 1000000
    i2 = 1000000
    if edges[edgeid_x][0] in traja:
        i1 = np.where(traja==edges[edgeid_x][0])[0][0]
    if edges[edgeid_x][1] in traja:
        i2 = np.where(traja==edges[edgeid_x][1])[0][0]
    i = min(i1,i2)
    if i==i1:
        pstime = i1+proj_val_x
    else:
        pstime = i1-proj_val_x
    return pstime

def pseudo_time_trajectory(traj,ProjStruct):
    projval = ProjStruct['ProjectionValues']
    edgeid = (ProjStruct['EdgeID']).astype(int)
    edges = ProjStruct['Edges']
    partition = ProjStruct['Partition']
    traj_points = np.zeros(0,'int32')
    for p in traj:
        traj_points = np.concatenate((traj_points,np.where(partition==p)[0]))
    #print(len(traj_points))
    pst = np.zeros(len(traj_points))
    for i,p in enumerate(traj_points):
        pst[i] = pseudo_time(traj[0],p,traj,projval,edgeid,edges)
    return pst,traj_points

def project_on_tree(X,tree):
    nodep = tree['NodePositions']
    edges = tree['Edges'][0]
    partition, dists = elpigraph.src.core.PartitionData(X = X, NodePositions = nodep, MaxBlockSize = 100000000, TrimmingRadius = np.inf,SquaredX = np.sum(X**2,axis=1,keepdims=1))
    ProjStruct = elpigraph.src.reporting.project_point_onto_graph(X = X,
                                     NodePositions = nodep,
                                     Edges = edges,
                                     Partition = partition)
    #projval = ProjStruct['ProjectionValues']
    #edgeid = (ProjStruct['EdgeID']).astype(int)
    ProjStruct['Partition'] = partition
    return ProjStruct

def quantify_pseudotime(all_trajectories,ProjStruct,producePlot=False):
    projval = ProjStruct['ProjectionValues']
    edgeid = (ProjStruct['EdgeID']).astype(int)
    edges = ProjStruct['Edges']
    partition = ProjStruct['Partition']
    PseudoTimeTraj = []
    for traj in all_trajectories:
        pst,points = pseudo_time_trajectory(traj,ProjStruct)
        pstt = {}
        pstt['Trajectory'] = traj
        pstt['Points'] = points
        pstt['Pseudotime'] = pst
        PseudoTimeTraj.append(pstt)
        if producePlot:
            plt.plot(np.sort(pst))
    return PseudoTimeTraj

def regress_variable_on_pseudotime(pseudotime,vals,TrajName,var_name,var_type,producePlot=True,verbose=False,Continuous_Regression_Type='linear',R2_Threshold=0.5,max_sample=-1,alpha_factor=2):
    # Continuous_Regression_Type can be 'linear','gpr' for Gaussian Process, 'kr' for kernel ridge
    if var_type=='BINARY':
        #convert back to binary vals
        mn = min(vals)
        mx = max(vals)
        vals[np.where(vals==mn)] = 0
        vals[np.where(vals==mx)] = 1
        if len(np.unique(vals))==1:
            regressor = None
        else:
            regressor = LogisticRegression(random_state=0,max_iter=1000,penalty='none').fit(pseudotime, vals)
    if var_type=='CATEGORICAL':
        if len(np.unique(vals))==1:
            regressor = None
        else:
            regressor = LogisticRegression(random_state=0,max_iter=1000,penalty='none').fit(pseudotime, vals)
    if var_type=='CONTINUOUS' or var_type=='ORDINAL':
        if len(np.unique(vals))==1:
            regressor = None
        else:
            if Continuous_Regression_Type=='gpr':
                # subsampling if needed
                pst = pseudotime.copy()
                vls = vals.copy()
                if max_sample>0:
                    l = list(range(len(vals)))
                    random.shuffle(l)
                    index_value = random.sample(l, min(max_sample,len(vls)))
                    pst = pst[index_value]
                    vls = vls[index_value]
                if len(np.unique(vls))>1:
                     gp_kernel =  C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
                     #gp_kernel =  RBF(np.std(vals))
                     regressor = GaussianProcessRegressor(kernel=gp_kernel,alpha=np.var(vls)*alpha_factor)
                     regressor.fit(pst, vls)
                else:
                     regressor = None
            if Continuous_Regression_Type=='linear':
                regressor = LinearRegression()
                regressor.fit(pseudotime, vals)
    
    r2score = 0
    if regressor is not None:
        r2score = r2_score(vals,regressor.predict(pseudotime))
    
        if producePlot and r2score>R2_Threshold:
            plt.plot(pseudotime,vals,'ro',label='data')
            unif_pst = np.linspace(min(pseudotime),max(pseudotime),100)
            pred = regressor.predict(unif_pst)
            if var_type=='BINARY' or var_type=='CATEGORICAL':
                prob = regressor.predict_proba(unif_pst)
                plt.plot(unif_pst,prob[:,1],'g-',linewidth=2,label='proba')    
            if var_type=='CONTINUOUS' or var_type=='ORDINAL':
                plt.plot(unif_pst,pred,'g-',linewidth=2,label='predicted')
            bincenters,wav = moving_weighted_average(pseudotime,vals.reshape(-1,1),step_size=1.5)
            plt.plot(bincenters,fill_gaps_in_number_sequence(wav),'b-',linewidth=2,label='sliding av')
            plt.xlabel('Pseudotime',fontsize=20)
            plt.ylabel(var_name,fontsize=20)
            plt.title(TrajName+', r2={:2.2f}'.format(r2score),fontsize=20)
            plt.legend(fontsize=15)
            plt.show()

    
    return r2score, regressor

def regression_of_variable_with_trajectories(PseudoTimeTraj,var,var_names,variable_types,X_original,verbose=False,producePlot=True,R2_Threshold=0.5,Continuous_Regression_Type='linear',max_sample=1000,alpha_factor=2):
    List_of_Associations = []
    for i,pstt in enumerate(PseudoTimeTraj):
        inds = pstt['Trajectory']
        #traj_nodep = nodep_original[inds,:]
        points = pstt['Points']
        pst = pstt['Pseudotime']
        pst = pst.reshape(-1,1)
        TrajName = 'Trajectory:'+str(pstt['Trajectory'][0])+'--'+str(pstt['Trajectory'][-1])
        k = var_names.index(var)
        vals = X_original[points,k]
        #print(np.mean(vals))
        r2,regressor = regress_variable_on_pseudotime(pst,vals,TrajName,var,variable_types[k],producePlot=producePlot,verbose=verbose,R2_Threshold=R2_Threshold,Continuous_Regression_Type=Continuous_Regression_Type, max_sample=max_sample,alpha_factor=alpha_factor)
        print(var,r2)
        pstt[var+'_regressor'] = regressor
        asstup = (TrajName,var,r2)
        #if verbose:
        #    print(var,'R2',r2)
        if r2>R2_Threshold:
            List_of_Associations.append(asstup)
            if verbose:
                print(i,asstup)
    return List_of_Associations

def moving_weighted_average(x, y, step_size=.1, steps_per_bin=1,
                            weights=None):
    # This ensures that all samples are within a bin
    number_of_bins = int(np.ceil(np.ptp(x) / step_size))
    bins = np.linspace(np.min(x), np.min(x) + step_size*number_of_bins,
                       num=number_of_bins+1)
    bins -= (bins[-1] - np.max(x)) / 2
    bin_centers = bins[:-steps_per_bin] + step_size*steps_per_bin/2

    counts, _ = np.histogram(x, bins=bins)
    #print(bin_centers)
    #print(counts)
    vals, _ = np.histogram(x, bins=bins, weights=y)
    bin_avgs = vals / counts
    #print(bin_avgs)
    n = len(bin_avgs)
    windowed_bin_avgs = as_strided(bin_avgs,
                                   (n-steps_per_bin+1, steps_per_bin),
                                   bin_avgs.strides*2)
    
    weighted_average = np.average(windowed_bin_avgs, axis=1, weights=weights)
    return bin_centers, weighted_average

def fill_gaps_in_number_sequence(x):
    firstnonnan,val = firstNonNan(x)
    firstnan = firstNanIndex(x)
    if firstnan is not None:
        x[0:firstnonnan] = val
    lastnonnan,val = lastNonNan(x)
    if firstnan is not None:
        x[lastnonnan:-1] = val
        x[-1] = val
    #print('Processing',x)
    firstnan = firstNanIndex(x)
    while firstnan is not None:
        #print(x[firstNanIndex:])
        firstnonnan,val = firstNonNan(x[firstnan:])
        #print(val)
        firstnonnan = firstnonnan+firstnan
        #print('firstNanIndex',firstnan)
        #print('firstnonnan',firstnonnan)
        #print(np.linspace(x[firstnan-1],val,firstnonnan-firstnan+2))
        x[firstnan-1:firstnonnan+1] = np.linspace(x[firstnan-1],val,firstnonnan-firstnan+2)
        #print('Imputed',x)
        firstnan = firstNanIndex(x)
    return x

def firstNonNan(floats):
  for i,item in enumerate(floats):
    if math.isnan(item) == False:
      return i,item

def firstNanIndex(floats):
  for i,item in enumerate(floats):
    if math.isnan(item) == True:
      return i

def lastNonNan(floats):
  for i,item in enumerate(np.flip(floats)):
    if math.isnan(item) == False:
      return len(floats)-i-1,item

def kernel_regressor(x,y,max_sample=1000,alpha_factor=2):          
    pst = x
    vls = y
    if max_sample>0:
        l = list(range(len(vls)))
        random.shuffle(l)
        index_value = random.sample(l, min(max_sample,len(vls)))
        pst = pst[index_value]
        vls = vls[index_value]
    regressor = None
    if len(np.unique(vls))>1:
        gp_kernel =  C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2))
        regressor = GaussianProcessRegressor(kernel=gp_kernel,alpha=np.var(vls)*alpha_factor)
        regressor.fit(pst, vls)
    unif_pst = None
    pred = None
    if not regressor==None:
        unif_pst = np.linspace(min(pst),max(pst),100)
        pred = regressor.predict(unif_pst)
    return unif_pst, pred

import networkx as nx
import random

def visualize_eltree_with_data(tree_elpi,X,X_original,principal_component_vectors,mean_vector,color,variable_names,
                              showEdgeNumbers=False,showNodeNumbers=False,showBranchNumbers=False,showPointNumbers=False,
                              Color_by_feature = '', Feature_Edge_Width = '', Invert_Edge_Value = False,
                              Min_Edge_Width = 5, Max_Edge_Width = 5, 
                              Big_Point_Size = 100, Small_Point_Size = 1, Normal_Point_Size = 20,
                              Visualize_Edge_Width_AsNodeCoordinates=True,
                              Color_by_partitioning = False,
                              visualize_partition = [],
                              Transparency_Alpha = 0.2,
                              Transparency_Alpha_points = 1,
                              verbose=False,
                              Visualize_Branch_Class_Associations = [], #list_of_branch_class_associations
                              cmap = 'cool',scatter_parameter=0.03,highlight_subset=[],
                              add_color_bar=False,
                              vmin=-1,vmax=-1,
                              percentile_contraction=20):

    nodep = tree_elpi['NodePositions']
    nodep_original = np.matmul(nodep,principal_component_vectors[:,0:X.shape[1]].T)+mean_vector
    adjmat = tree_elpi['ElasticMatrix']
    edges = tree_elpi['Edges'][0]
    color2 = color
    if not Color_by_feature=='':
        k = variable_names.index(Color_by_feature)
        color2 = X_original[:,k]
    if Color_by_partitioning:
        color2 = visualize_partition
        color_seq = [[1,0,0],[0,1,0],[0,0,1],[0,1,1],[1,0,1],[1,1,0],
             [1,0,0.5],[1,0.5,0],[0.5,0,1],[0.5,1,0],
             [0.5,0.5,1],[0.5,1,0.5],[1,0.5,0.5],
             [0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0],[0.5,0.5,0.5],[0,0,0.5],[0,0.5,0],[0.5,0,0],
             [0,0.25,0.5],[0,0.5,0.25],[0.25,0,0.5],[0.25,0.5,0],[0.5,0,0.25],[0.5,0.25,0],
             [0.25,0.25,0.5],[0.25,0.5,0.25],[0.5,0.25,0.25],[0.25,0.25,0.5],[0.25,0.5,0.25],
             [0.25,0.25,0.5],[0.25,0.5,0.25],[0.5,0,0.25],[0.5,0.25,0.25]]
        color2_unique, color2_count = np.unique(color2, return_counts=True)
        inds = sorted(range(len(color2_count)), key=lambda k: color2_count[k], reverse=True)
        newc = []
        for i,c in enumerate(color2):
            k = np.where(color2_unique==c)[0][0]
            count = color2_count[k]
            k1 = np.where(inds==k)[0][0]
            k1 = k1%len(color_seq)
            col = color_seq[k1]
            newc.append(col)
        color2 = newc
    
    plt.style.use('ggplot')
    points_size = Normal_Point_Size*np.ones(X_original.shape[0])
    if len(Visualize_Branch_Class_Associations)>0:
        points_size = Small_Point_Size*np.ones(X_original.shape[0])
        for assc in Visualize_Branch_Class_Associations:
            branch = assc[0]
            cls = assc[1]
            indices = [i for i, x in enumerate(color) if x == cls]
            #print(branch,cls,color,np.where(color==cls))
            points_size[indices] = Big_Point_Size

    node_size = 10
    #Associate each node with datapoints
    if verbose:
        print('Partitioning the data...')
    partition, dists = elpigraph.src.core.PartitionData(X = X, NodePositions = nodep, MaxBlockSize = 100000000, TrimmingRadius = np.inf,SquaredX = np.sum(X**2,axis=1,keepdims=1))
    #col_nodes = {node: color[np.where(partition==node)[0]] for node in np.unique(partition)}


    #Project points onto the graph
    if verbose:
        print('Projecting data points onto the graph...')
    ProjStruct = elpigraph.src.reporting.project_point_onto_graph(X = X,
                                     NodePositions = nodep,
                                     Edges = edges,
                                     Partition = partition)

    projval = ProjStruct['ProjectionValues']
    edgeid = (ProjStruct['EdgeID']).astype(int)
    X_proj = ProjStruct['X_projected']

    dist2proj = np.sum(np.square(X-X_proj),axis=1)
    shift = np.percentile(dist2proj,percentile_contraction)
    dist2proj = dist2proj-shift

    #Create graph
    if verbose:
        print('Producing graph layout...')
    g=nx.Graph()
    g.add_edges_from(edges)
    pos = nx.kamada_kawai_layout(g,scale=2)
    #pos = nx.planar_layout(g)
    #pos = nx.spring_layout(g,scale=2)
    idx=np.array([pos[j] for j in range(len(pos))])

    #plt.figure(figsize=(16,16))
    if verbose:
        print('Calculating scatter aroung the tree...')
    x = np.zeros(len(X))
    y = np.zeros(len(X))
    for i in range(len(X)):
        # distance from edge
	# This is squared distance from a node
        #r = np.sqrt(dists[i])*scatter_parameter
	# This is squared distance from a projection (from edge),
	# even though the difference might be tiny
        r = 0
        if dist2proj[i]>0:
            r = np.sqrt(dist2proj[i])*scatter_parameter        

        #get node coordinates for this edge
        x_coos = np.concatenate((idx[edges[edgeid[i],0],[0]],idx[edges[edgeid[i],1],[0]]))
        y_coos = np.concatenate((idx[edges[edgeid[i],0],[1]],idx[edges[edgeid[i],1],[1]]))

        projected_on_edge = False
  
        if projval[i]<0:
            #project to 0% of the edge (first node)
            x_coo = x_coos[0] 
            y_coo = y_coos[0]
        elif projval[i]>1: 
            #project to 100% of the edge (second node)
            x_coo = x_coos[1]
            y_coo = y_coos[1]
        else:   
            #project to appropriate % of the edge
            x_coo = x_coos[0] + (x_coos[1]-x_coos[0])*projval[i]
            y_coo = y_coos[0] + (y_coos[1]-y_coos[0])*projval[i]
            projected_on_edge = True

        #if projected_on_edge:
        #     color2[i]=0
        #else:
        #     color2[i]=1    
        #random angle
        #alpha = 2 * np.pi * np.random.random()
        #random scatter to appropriate distance 
        #x[i] = r * np.cos(alpha) + x_coo
        #y[i] = r * np.sin(alpha) + y_coo
	# we rather position the point close to project and put
	# it at distance r orthogonally to the edge 
	# on a random side of the edge 
        # However, if projection was on a node then we scatter 
        # in random direction
        vex = x_coos[1]-x_coos[0]
        vey = y_coos[1]-y_coos[0]
        if not projected_on_edge:
            vex = np.random.random()-0.5
            vey = np.random.random()-0.5
        vn = np.sqrt(vex*vex+vey*vey)
        vex = vex/vn
        vey = vey/vn
        rsgn = random_sign()
        x[i] = x_coo+vey*r*rsgn
        y[i] = y_coo-vex*r*rsgn
    if vmin<0:
        vmin=min(color2)
    if vmax<0:
        vmax=max(color2)
    plt.scatter(x,y,c=color2,cmap=cmap,s=points_size, vmin=vmin, vmax=vmax,alpha=Transparency_Alpha_points)
    if showPointNumbers:
        for j in range(len(X)):
            plt.text(x[j],y[j],j)
    if len(highlight_subset)>0:
        color_subset = [color2[i] for i in highlight_subset]
        plt.scatter(x[highlight_subset],y[highlight_subset],c=color_subset,cmap=cmap,s=Big_Point_Size)
    if add_color_bar:
        plt.colorbar()


    #Scatter nodes
    tree_elpi['NodePositions2D'] = idx
    plt.scatter(idx[:,0],idx[:,1],s=node_size,c='black',alpha=.8)

    #Associate edge width to a feature
    edge_vals = [1]*len(edges)
    if not Feature_Edge_Width=='' and not Visualize_Edge_Width_AsNodeCoordinates:
        k = variable_names.index(Feature_Edge_Width)
        for j in range(len(edges)):
            vals = X_original[np.where(edgeid==j)[0],k]
            vals = (np.array(vals)-np.min(X_original[:,k]))/(np.max(X_original[:,k])-np.min(X_original[:,k]))
            edge_vals[j] = np.mean(vals)
        for j in range(len(edges)):
            if np.isnan(edge_vals[j]):
                e = edges[j]
                inds = [ei for ei,ed in enumerate(edges) if ed[0]==e[0] or ed[1]==e[0] or ed[0]==e[1] or ed[1]==e[1]]
                inds.remove(j)
                evals = np.array(edge_vals)[inds]
                #print(j,inds,evals,np.mean(evals))
                edge_vals[j] = np.mean(evals[~np.isnan(evals)])
        if Invert_Edge_Value:
            edge_vals = [1-v for v in edge_vals]

    if not Feature_Edge_Width=='' and Visualize_Edge_Width_AsNodeCoordinates:
        k = variable_names.index(Feature_Edge_Width)
        for j in range(len(edges)):
            e = edges[j]
            amp = np.max(nodep_original[:,k])-np.min(nodep_original[:,k])
            mn = np.min(nodep_original[:,k])
            v0 = (nodep_original[e[0],k]-mn)/amp
            v1 = (nodep_original[e[1],k]-mn)/amp
            #print(v0,v1)
            edge_vals[j] = (v0+v1)/2
        if Invert_Edge_Value:
            edge_vals = [1-v for v in edge_vals]
        
    #print(edge_vals)
            
    
    #Plot edges
    for j in range(len(edges)):
        x_coo = np.concatenate((idx[edges[j,0],[0]],idx[edges[j,1],[0]]))
        y_coo = np.concatenate((idx[edges[j,0],[1]],idx[edges[j,1],[1]]))
        plt.plot(x_coo,y_coo,c='k',linewidth=Min_Edge_Width+(Max_Edge_Width-Min_Edge_Width)*edge_vals[j],alpha=Transparency_Alpha)
        if showEdgeNumbers:
            plt.text((x_coo[0]+x_coo[1])/2,(y_coo[0]+y_coo[1])/2,j,FontSize=20,bbox=dict(facecolor='grey', alpha=0.5))

    if showBranchNumbers:
        branch_vals = list(set(visualize_partition))
        for i,val in enumerate(branch_vals):
            ind = visualize_partition==val
            xbm = np.mean(x[ind])
            ybm = np.mean(y[ind])
            plt.text(xbm,ybm,int(val),FontSize=20,bbox=dict(facecolor='grey', alpha=0.5))
        
    if showNodeNumbers:
        for i in range(nodep.shape[0]):
            plt.text(idx[i,0],idx[i,1],str(i),FontSize=20,bbox=dict(facecolor='grey', alpha=0.5))

    #plt.axis('off')
    coords = np.zeros((len(x),2))
    coords[:,0] = x
    coords[:,1] = y
    return coords

def random_sign():
    return 1 if random.random() < 0.5 else -1
