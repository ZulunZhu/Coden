#include "instantAlg.h"
#include "Graph.h"
#include <sys/stat.h>
#include <sys/types.h>
using namespace std;
using namespace Eigen;
#include <set>
namespace propagation
{

vector<vector<uint>> Instantgnn::update_graph(string updatefilename, vector<uint>&affected_nodelst, vector<vector<uint>>&delete_neighbors) // vector<vector<uint>>&add_adjs
{
    ifstream infile(updatefilename.c_str());
    
    if(!infile.is_open()){
        cerr << " Update file doesn't exist! " << endl;
        exit(EXIT_FAILURE);
    }
    //cout<<"updating graph " << updatefilename <<endl;
    uint v_from, v_to;
    int insertFLAG = 0;

    vector<vector<uint>> new_neighbors(vert);
    vector<bool> isAffected(vert, false);
    while (infile >> v_from >> v_to)
    {
        insertFLAG = g.isEdgeExist(v_from, v_to);
        // update graph
        if(!isAffected[v_from]){
            affected_nodelst.push_back(v_from);
            isAffected[v_from] = true;
        }
        
        if(insertFLAG == 1){
            g.insertEdge(v_from, v_to);
            new_neighbors[v_from].push_back(v_to);
        }
        else if(insertFLAG == -1){
            // cout<<"delete......"<<endl;
            // g.deleteEdge(v_from, v_to);
            // delete_neighbors[v_from].push_back(v_to);
        }
    }
    infile.close();

    cout<<"update graph finish..."<<"affected_nodelst.size():"<<affected_nodelst.size()<<endl;

    return new_neighbors;
}

bool err_cmp_pos(const pair<int,double> a,const pair<int,double> b){
	return a.second > b.second;
}
bool err_cmp_neg(const pair<int,double> a,const pair<int,double> b){
	return a.second < b.second;
}
string remove_extension(const string& filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == string::npos) return filename; // no extension found
    return filename.substr(0, lastdot);
}
void split_edges(const string& input_file, const vector<size_t>& indices) {
    ifstream infile(input_file);
    if (!infile) {
        cerr << "Error opening input file: " << input_file << endl;
        return;
    }

    string output_dir = remove_extension(input_file);
    // Create the output directory if it doesn't exist
    if (mkdir(output_dir.c_str(), 0777) == -1) {
        if (errno != EEXIST) {
            cerr << "Error creating directory: " << output_dir << endl;
            return;
        }
    }

    string line;
    vector<ofstream> outfiles;
    size_t current_index = 0;

    // Initialize output file streams
    for (size_t i = 0; i <= indices.size(); ++i) {
        string outfile_name = output_dir + "/edges_part_" + to_string(i) + ".txt";
        outfiles.emplace_back(outfile_name);
        if (!outfiles.back()) {
            cerr << "Error opening output file: " << outfile_name << endl;
            return;
        }
    }

    size_t line_num = 0;
    size_t file_index = 0;

    while (getline(infile, line)) {
        if (file_index < indices.size() && line_num == indices[file_index]) {
            ++file_index;
        }
        outfiles[file_index] << line << endl;
        ++line_num;
    }

    // Close all output files
    for (auto& outfile : outfiles) {
        outfile.close();
    }

    infile.close();
}

void Instantgnn::split_batch(string updatefilename, double epsilon, Eigen::Map<Eigen::MatrixXd> &x_max, double x_norm)
{
    dimension=x_max.cols();
    cout<<"dimension: "<<dimension<<", row:"<<x_max.rows()<<"x_norm"<<x_norm<<endl;
    cout<<"x_max(0,335)"<<x_max(335,0)<<endl;

    ifstream infile(updatefilename.c_str());


    if(!infile.is_open()){
        cerr << " Update file doesn't exist! " << endl;
        exit(EXIT_FAILURE);
    }
    //cout<<"updating graph " << updatefilename <<endl;
    uint v_from, v_to;
    int insertFLAG = 0;

    double error = 0;
    double error_max = epsilon*x_norm;
    uint indice = 0;
    std::vector<size_t> split_indices;
    vector<vector<uint>> new_neighbors(vert);
    vector<bool> isAffected(vert, false);
    while (infile >> v_from >> v_to)
    {
        indice++;
        double L_norm = 0;
        insertFLAG = g_copy.isEdgeExist(v_from, v_to);
        // update graph
        
        if(insertFLAG == 1){
            
            uint outSize = g_copy.getOutSize(v_from);
            for(uint i=0; i<outSize; i++)
            {
                uint v = g_copy.getOutVert(v_from, i);
                L_norm = L_norm+ 1.0/ (outSize*(outSize+1))*x_max(v,0);

            }
            L_norm =L_norm+1.0/ (outSize+1)*x_max(v_to,0);
            g_copy.insertEdge(v_from, v_to);
            
        }
        else if(insertFLAG == -1){
            
            // cout<<"delete......"<<endl;
            // exit(0);
            g_copy.deleteEdge(v_from, v_to);
            uint outSize = g_copy.getOutSize(v_from);
            for(uint i=0; i<outSize; i++)
            {
                uint v = g_copy.getOutVert(v_from, i);
                L_norm = L_norm+ 1.0/ (outSize*(outSize+1))*x_max(v,0);

            }
            L_norm =L_norm+1.0/ (outSize+1)*x_max(v_to,0);
        }        

        error = error+ L_norm;
        
        // cout<<"error_max "<<error_max<<"L_norm "<<L_norm<<"error:"<<error<<"indice"<<indice<<endl;

        if(error>error_max){
            cout<<"indice"<< indice<<endl;
            split_indices.push_back(indice);
            error = 0;
        }
    }
    cout<<"sum of updated edges:"<< indice<<endl;
    
    infile.close();
    std::cout << "Indices: ";
    for (int index : split_indices) {
        std::cout << index << " ";
    }
    std::cout << std::endl;
    split_edges(updatefilename, split_indices);

    
}
//batch_lazy_update
void Instantgnn::snapshot_lazy(string updatefilename, double rmaxx,double rbmax, double delta,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat_t, Eigen::Map<Eigen::MatrixXd> &change_node_list, string algorithm)
{
    alpha=alphaa;
    rmax=rmaxx;
    bool dimension_flag = false;
    Eigen::MatrixXd feat;
    if(feat_t.cols()>feat_t.rows()){
        cout<<" Fault with the feature dimension!"<<endl;
        dimension_flag = true;
        feat  = feat_t.transpose();
        
    }else{
        feat  = feat_t;
    }
    
    dimension=feat.cols();
    cout<<"dimension: "<<dimension<<", col:"<<feat.rows()<<endl;
    vector<queue<uint>> candidate_sets(dimension);
    vector<vector<bool>> isCandidates(dimension, vector<bool>(vert, false));
    vector<bool> isUpdateW(dimension, false);

    clock_t start_t, end_t;
    start_t = clock();
    cout<<"Lazy updating begin, for snapshot: " << updatefilename <<endl;
    
    //update graph, obtain affected node_list
    vector<uint> affected_nodelst;

    //reverse push, get the affected node_list really needed to push

    vector<uint> changed_nodes;
    vector<vector<uint>> delete_neighbors(vert);
    vector<vector<uint>> add_neighbors(vert);

    add_neighbors = update_graph(updatefilename, affected_nodelst, delete_neighbors);
    end_t = clock();
    //cout<<"-----update_graph finish-------- time: " << (end_t - start_t)/(1.0*CLOCKS_PER_SEC)<<" s"<<endl;
    //cout<<"affected_nodelst.size():"<<affected_nodelst.size()<<endl;

    //deal nodes in affected node_list, update \pi and r
    vector<double> oldDu(affected_nodelst.size(), 0);
    //double oldDu[affected_nodelst.size()];


    // cout<<'------------------ reverse push -------------------'<<endl;
    double n = feat.rows();
    // cout<<"****number of nodes"<<n <<endl;

    double errorlimit=1.0/n;
    double epsrate=1;
    

    // my code ***********************************************************************************
//     for(uint i=0;i<affected_nodelst.size();i++)
//     {
//         uint affected_node = affected_nodelst[i];
//         // update Du
//         oldDu[i] = Du[affected_node]; //[d(u)-delta_d(u)]^0.5
//         Du[affected_node] = pow(g.getOutSize(affected_node), 0.5);
        
//     }
//     // MSG(feat(36,36));
//     clock_t end_t2 = clock();
//     //cout << "-----update pi and r finish----- time: "<< (end_t2 - end_t)/(1.0*CLOCKS_PER_SEC)<<" s" <<endl;
    
//    // *******************Update the embedding***********************//
//     // reset r
//     R = vector<vector<double>>(dimension, vector<double>(vert, 0));
//     //update r
//     for(uint i=0; i<affected_nodelst.size(); i++)
//     {
//         uint affected_node = affected_nodelst[i];
//         for(int dim=0; dim<dimension; dim++)
//         {
//             double rowsum_p=rowsum_pos[dim];
//             double rowsum_n=rowsum_neg[dim];
//             double rmax_p=rowsum_p*rmax;
//             double rmax_n=rowsum_n*rmax;
//             double increment = feat(affected_node,dim)*(1-alpha)/Du[affected_node];
//             for(uint j=0; j<add_neighbors[affected_node].size(); j++)
//             {
//                 uint add_node = add_neighbors[affected_node][j];
//                 R[dim][add_node] += increment/Du[add_node];
//                 if( R[dim][add_node]>rmax_p || R[dim][add_node]<rmax_n )
//                 {
//                     if(!isCandidates[dim][add_node]){
//                         candidate_sets[dim].push(add_node);
//                         isCandidates[dim][add_node] = true;
//                     }
//                     if(!isUpdateW[dim]){
//                     update_w.push_back(dim);
//                     isUpdateW[dim] = true;
//                     }
//                 }
//             }
//             for(uint j=0; j<delete_neighbors[affected_node].size(); j++)
//             {
//                 uint delete_node = delete_neighbors[affected_node][j];
//                 R[dim][delete_node] -= increment/Du[delete_node];
//                 if( R[dim][delete_node]>rmax_p || R[dim][delete_node]<rmax_n )
//                 {
//                     if(!isCandidates[dim][delete_node]){
//                         candidate_sets[dim].push(delete_node);
//                         isCandidates[dim][delete_node] = true;
//                     }
//                     if(!isUpdateW[dim]){
//                     update_w.push_back(dim);
//                     isUpdateW[dim] = true;
//                     }
//                 }
//             }
//             std::set<int> tempSet;
//             std::set<int> finalSet;
//             // Perform outAdj - add_neighbors
//             std::set_difference(
//                 g.outAdj[affected_node].begin(), g.outAdj[affected_node].end(),
//                 add_neighbors[affected_node].begin(), add_neighbors[affected_node].end(),
//                 std::inserter(tempSet, tempSet.begin())
//             );

//             // Perform intermediate result - delete_neighbors
//             std::set_difference(
//                 tempSet.begin(), tempSet.end(),
//                 delete_neighbors[affected_node].begin(), delete_neighbors[affected_node].end(),
//                 std::inserter(finalSet, finalSet.begin())
//             );
//             // Convert the final set back to a vector
//             std::vector<int> result(finalSet.begin(), finalSet.end());

//             // for (int add_node : result) 
//             // {
//             //     R[dim][add_node] += increment*(1.0/Du[affected_node]- 1.0/oldDu[affected_node]);
//             //     if( R[dim][add_node]>rmax_p || R[dim][add_node]<rmax_n )
//             //     {
//             //         if(!isCandidates[dim][add_node]){
//             //             candidate_sets[dim].push(add_node);
//             //             isCandidates[dim][add_node] = true;
//             //         }
//             //         if(!isUpdateW[dim]){
//             //         update_w.push_back(dim);
//             //         isUpdateW[dim] = true;
//             //         }
//             //     }
//             // }
            

//         }
//     }
    // end of my code ***********************************************************************************
    //original code*******************************************************************
    for(uint i=0;i<affected_nodelst.size();i++)
    {
        uint affected_node = affected_nodelst[i];
        // update Du
        oldDu[i] = Du[affected_node]; //[d(u)-delta_d(u)]^0.5
        Du[affected_node] = pow(g.getOutSize(affected_node), 0.5);
        
        //update \pi(u) to avoid dealing with N(u), r needs to be updated accordingly
        for(int dim=0; dim<dimension; dim++)
        {
            feat(affected_node,dim) = feat(affected_node,dim) * Du[affected_node] / oldDu[i];
            double delta_1 = feat(affected_node,dim) * (oldDu[i]-Du[affected_node]) / alpha / Du[affected_node];
            R[dim][affected_node] += delta_1;
        }
    }
    // MSG(feat(36,36));
    clock_t end_t2 = clock();
    //cout << "-----update pi and r finish----- time: "<< (end_t2 - end_t)/(1.0*CLOCKS_PER_SEC)<<" s" <<endl;
  
    //update r
    for(uint i=0; i<affected_nodelst.size(); i++)
    {
        uint affected_node = affected_nodelst[i];
        for(int dim=0; dim<dimension; dim++)
        {
            double rowsum_p=rowsum_pos[dim];
            double rowsum_n=rowsum_neg[dim];
            double rmax_p=rowsum_p*rmax;
            double rmax_n=rowsum_n*rmax;
            double increment = feat(affected_node,dim)*alpha/Du[affected_node];
            for(uint j=0; j<add_neighbors[affected_node].size(); j++)
            {
                uint add_node = add_neighbors[affected_node][j];
                R[dim][add_node] += increment;
            }
            for(uint j=0; j<delete_neighbors[affected_node].size(); j++)
            {
                uint delete_node = delete_neighbors[affected_node][j];
                R[dim][delete_node] -= increment;
            }

            if( R[dim][affected_node]>rmax_p || R[dim][affected_node]<rmax_n )
            {
                if(!isCandidates[dim][affected_node]){
                    candidate_sets[dim].push(affected_node);
                    isCandidates[dim][affected_node] = true;
                }
                if(!isUpdateW[dim]){
                    update_w.push_back(dim);
                    isUpdateW[dim] = true;
                }
            }
        }
    }
    
    clock_t end_t3 = clock();
    //cout<<"-----update r finish----- time: "<<(end_t3 - end_t2)/(1.0*CLOCKS_PER_SEC)<<" s" <<endl;
    

    for(int i=0; i<dimension; i++)
    {
        double rowsum_p=rowsum_pos[i];
        double rowsum_n=rowsum_neg[i];
        double rmax_p=rowsum_p*rmax;
        double rmax_n=rowsum_n*rmax;// to scale with the feature
        for(uint j=0; j<vert; j++)
        {
            if(R[i][j]>0)
                R_sum_pos[i]+=R[i][j];
                if(R[i][j]>R_max_pos[i])
                    R_max_pos[i] = R[i][j];
            else
                R_sum_neg[i]+=R[i][j];
                if(R[i][j]<R_max_neg[i])
                    R_max_neg[i] = R[i][j];
            // if(R[i][j]>rmax_p || R[i][j]<rmax_n)
            // {   
            //     if(isCandidates[i][j]==false){
            //         candidate_sets[i].push(j);
            //         isCandidates[i][j] = true;
            //     }
            // }
        }
        
        
    }
    //push
    if(update_w.size()>0)
    {
        cout<<"dims of feats that need push:"<<update_w.size()<<endl;

        cout<<"before push feat(36,36):"<<feat(36,36)<<endl;
        Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,true,algorithm,false, change_node_list);
        cout<<"after push feat(36,36):"<<feat(36,36)<<endl;
    }

    if(dimension_flag == true){
        feat_t  = feat.transpose();
    }
    
 
    

}




//batch_update
void Instantgnn::snapshot_operation(string updatefilename, double rmaxx,double alphaa, Eigen::Map<Eigen::MatrixXd> &feat, string algorithm)
{
    alpha=alphaa;
    rmax=rmaxx;

    vector<queue<uint>> candidate_sets(dimension);
    vector<vector<bool>> isCandidates(dimension, vector<bool>(vert, false));
    vector<bool> isUpdateW(dimension, false);

    clock_t start_t, end_t;
    start_t = clock();
    cout<<"updating begin, for snapshot: " << updatefilename <<endl;
    cout<<"******************************** "<<endl;
    //update graph, obtain affected node_list
    vector<uint> affected_nodelst;
    vector<pair<uint,uint>> affected_node_pair;
    vector<vector<uint>> delete_neighbors(vert);
    vector<vector<uint>> add_neighbors(vert);

    add_neighbors = update_graph(updatefilename, affected_nodelst, delete_neighbors);
    end_t = clock();
    //cout<<"-----update_graph finish-------- time: " << (end_t - start_t)/(1.0*CLOCKS_PER_SEC)<<" s"<<endl;
    //cout<<"affected_nodelst.size():"<<affected_nodelst.size()<<endl;

    //deal nodes in affected node_list, update \pi and r
    vector<double> oldDu(affected_nodelst.size(), 0);
    //double oldDu[affected_nodelst.size()];

    
    for(uint i=0;i<affected_nodelst.size();i++)
    {
        uint affected_node = affected_nodelst[i];
        // update Du
        oldDu[i] = Du[affected_node]; //[d(u)-delta_d(u)]^0.5
        Du[affected_node] = pow(g.getOutSize(affected_node), 0.5);
        
        //update \pi(u) to avoid dealing with N(u), r needs to be updated accordingly
        for(int dim=0; dim<dimension; dim++)
        {
            feat(affected_node,dim) = feat(affected_node,dim) * Du[affected_node] / oldDu[i];
            double delta_1 = feat(affected_node,dim) * (oldDu[i]-Du[affected_node]) / alpha / Du[affected_node];
            R[dim][affected_node] += delta_1;
        }
    }
    // MSG(feat(36,36));
    clock_t end_t2 = clock();
    //cout << "-----update pi and r finish----- time: "<< (end_t2 - end_t)/(1.0*CLOCKS_PER_SEC)<<" s" <<endl;
  
    //update r
    for(uint i=0; i<affected_nodelst.size(); i++)
    {
        uint affected_node = affected_nodelst[i];
        for(int dim=0; dim<dimension; dim++)
        {
            double rowsum_p=rowsum_pos[dim];
            double rowsum_n=rowsum_neg[dim];
            double rmax_p=rowsum_p*rmax;
            double rmax_n=rowsum_n*rmax;
            double increment = feat(affected_node,dim)*alpha/Du[affected_node];
            for(uint j=0; j<add_neighbors[affected_node].size(); j++)
            {
                uint add_node = add_neighbors[affected_node][j];
                R[dim][add_node] += increment;
            }
            for(uint j=0; j<delete_neighbors[affected_node].size(); j++)
            {
                uint delete_node = delete_neighbors[affected_node][j];
                R[dim][delete_node] -= increment;
            }

            if( R[dim][affected_node]>rmax_p || R[dim][affected_node]<rmax_n )
            {
                if(!isCandidates[dim][affected_node]){
                    candidate_sets[dim].push(affected_node);
                    isCandidates[dim][affected_node] = true;
                }
                if(!isUpdateW[dim]){
                    update_w.push_back(dim);
                    isUpdateW[dim] = true;
                }
            }
        }
    }
    clock_t end_t3 = clock();
    //cout<<"-----update r finish----- time: "<<(end_t3 - end_t2)/(1.0*CLOCKS_PER_SEC)<<" s" <<endl;
    double* data = nullptr;
    Eigen::Map<Eigen::MatrixXd> emptyMap(data,0, 0);
    //push
    if(update_w.size()>0)
    {
      cout<<"dims of feats that need push:"<<update_w.size()<<endl;
      if(algorithm == "instant"){
        cout<<"before push feat(36,36):"<<feat(36,36)<<endl;
        Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,true,algorithm,false, emptyMap);
        cout<<"after push feat(36,36):"<<feat(36,36)<<endl;
      }
      else if(algorithm == "speed_push"){
        cout<<"before push feat(36,36):"<<feat(36,36)<<endl;
        Instantgnn::ppr_push(update_w.size(), feat, false,candidate_sets,isCandidates,true,algorithm,false, emptyMap);
        cout<<"after push feat(36,36):"<<feat(36,36)<<endl;
        cout<<"speed push"<<endl;
      }

      
    }
}

int startsWith(string s, string sub){
        return s.find(sub)==0?1:0;
}
double Instantgnn::initial_operation(string path, string dataset,uint mm,uint nn,double rmaxx,double rbmax, double delta,double alphaa,double epsilonn, Eigen::Map<Eigen::MatrixXd> &feat_t, string algorithm)
{   
    // ppr.just_fortest();
    if(algorithm=="instant"){
        X = feat_t; // change in feat not influence X
    }
    bool dimension_flag = false;
    string filename = "./time.log";
    ofstream queryfile(filename, ios::app);
    queryfile<<"The dataset is = "<<dataset<<endl;
    queryfile.close();
    config.rbmax = rbmax/log(nn);
    config.delta = delta/nn;
    
    Eigen::MatrixXd feat;
    if(feat_t.cols()>feat_t.rows()){
        cout<<" Fault with the feature dimension!"<<endl;
        dimension_flag = true;
        feat  = feat_t.transpose();
        
    }else{
        feat  = feat_t;
    }
    
    dimension=feat.cols();
    cout<<"dimension: "<<dimension<<", col:"<<feat.rows()<<endl;
    
    
    // cout<<"C++ feat(357,37):"<<feat(357,37)<<endl;
    // cout<<"C++ feat(37,357):"<<feat(37,357)<<endl;
    // exit(0);


    dimension=feat.cols();
    cout<<"dimension: "<<dimension<<", col:"<<feat.rows()<<endl;
    dimension=min(feat.rows(),feat.cols());
    cout<<"dimension: "<<dimension<<", col:"<<max(feat.rows(),feat.cols())<<endl;
    
    rmax=rmaxx;
    edges=mm;
    vert=nn;
    alpha=alphaa;
    epsilon=epsilonn;
    omega = (2+epsilon)*log(2*vert)*vert/epsilon/epsilon;
    dataset_name=dataset;
    cout<<dataset_name<<endl;
  
    g.inputGraph(path, dataset_name, vert, edges);
    g_copy.inputGraph(path, dataset_name, vert, edges);
    // g.inputGraph_fromedgelist(path, dataset_name, vert, edges);
    Du=vector<double>(vert,0);
    double rrr=0.5;
    int c_m = 0;
    for(uint i=0; i<vert; i++)
    {   
        // c_m+= g.getOutSize(i);
        // cout<<" g.getOutSize(i) "<<c_m<<endl;
        Du[i]=pow(g.getOutSize(i),rrr);
    }

    R = vector<vector<double>>(dimension, vector<double>(vert, 0));
    rowsum_pos = vector<double>(dimension,0);
    rowsum_neg = vector<double>(dimension,0);
    R_sum_pos = vector<double>(dimension,0);
    R_sum_neg = vector<double>(dimension,0);
    R_max_pos = vector<double>(dimension,0);
    R_max_neg = vector<double>(dimension,0);
    random_w = vector<int>(dimension);

    

    for(int i = 0 ; i < dimension ; i++ )
        random_w[i] = i;
    random_shuffle(random_w.begin(),random_w.end());
    for(int i=0; i<dimension; i++)
    {
        for(uint j=0; j<vert; j++)
        {
            if(feat(j,i)>0)
                rowsum_pos[i]+=feat(j,i);
                if(feat(j,i)>R_max_pos[i])
                    R_max_pos[i] = feat(j,i);
            else
                rowsum_neg[i]+=feat(j,i);
                if(feat(j,i)<R_max_neg[i])
                    R_max_neg[i] = feat(j,i);
            
        }
        
        
    }
    double* data = nullptr;
    Eigen::Map<Eigen::MatrixXd> emptyMap(data,0, 0);
    vector<queue<uint>> candidate_sets(dimension);
    vector<vector<bool>> isCandidates(dimension, vector<bool>(vert, false));

    cout<<"before push feat(357,37):"<<feat(357,37)<<endl;
    Instantgnn::ppr_push(dimension, feat, true,candidate_sets,isCandidates,true,algorithm, false, emptyMap);
    cout<<"after push feat(357,37):"<<feat(357,37)<<endl;
    
    
    double dataset_size=(double)(((long long)edges+vert)*4+(long long)vert*dimension*8)/1024.0/1024.0/1024.0;
    if(dimension_flag == true){
        feat_t  = feat.transpose();
    }
    
    return dataset_size;
}

void Instantgnn::ppr_push(int dimension, Eigen::Ref<Eigen::MatrixXd>feat, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates, bool log, string algorithm, bool reverse, Eigen::Map<Eigen::MatrixXd> &change_node_list)
{
    vector<thread> threads;
    
    struct timeval t_start,t_end;
    double timeCost;
    //clock_t start_t, end_t;
    gettimeofday(&t_start,NULL);
    if(log)
        cout<<"Begin propagation..."<<init << "...dimension:"<< dimension <<endl;
        cout<<"candidate_sets:"<<candidate_sets[127].size()<<endl;
    int ti,start;
    int ends=0;
    
    //start_t = clock();
    for( ti=1 ; ti <= dimension%NUMTHREAD ; ti++ )
    {
        start = ends;
        ends+=ceil((double)dimension/NUMTHREAD);
        if(init)
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,true,std::ref(candidate_sets),std::ref(isCandidates),algorithm,reverse));
        else
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,false,std::ref(candidate_sets),std::ref(isCandidates),algorithm, reverse));
    }
    for( ; ti<=NUMTHREAD ; ti++ )
    {
        start = ends;
        ends+=dimension/NUMTHREAD;
        if(init)
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,true,std::ref(candidate_sets),std::ref(isCandidates),algorithm,reverse));
        else
            threads.push_back(thread(&Instantgnn::ppr_residue,this,feat,start,ends,false,std::ref(candidate_sets),std::ref(isCandidates),algorithm,reverse));
    }
    

    for (int t = 0; t < NUMTHREAD ; t++)
        threads[t].join();
    vector<thread>().swap(threads);
    update_w.clear();
    if(log)
        cout<<"remaining candidate_sets:"<<candidate_sets[127].size()<<endl;
    //end_t = clock();
    //double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    gettimeofday(&t_end, NULL);
    timeCost = t_end.tv_sec - t_start.tv_sec + (t_end.tv_usec - t_start.tv_usec)/1000000.0;
    if(log){
        cout<<"snap: "<<config.snap <<" time cost:"<<timeCost<<" s"<< " reverse ? ="<<reverse<<endl;
        //cout<<"The clock time : "<<total_t<<" s"<<endl;
    }
    if(!init){
        
        change_node_list(0) += timeCost;

    }
    
    string filename = "./time.log";
    ofstream queryfile(filename, ios::app);
    queryfile<<timeCost<<" ";
    queryfile.close();
    config.snap+=1;
    vector<vector<bool>>().swap(isCandidates);
    vector<queue<uint>>().swap(candidate_sets);


}
void Instantgnn::ppr_residue(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates, string algorithm, bool reverse)
{
    // string algorithm = "instant";
    // string algorithm = "speed_push";
    
    int w;
    for(int it=st;it<ed;it++)
    {
        if(reverse){
            w=1;
        }
        else if(init)
            w = random_w[it];
        else
            w = update_w[it];

        
        
        queue<uint> candidate_set = candidate_sets[w];
        vector<bool> isCandidate = isCandidates[w];

        double rowsum_p=rowsum_pos[w];
        double rowsum_n=rowsum_neg[w];
        double rmax_p=rowsum_p*rmax;
        double rmax_n=rowsum_n*rmax;// to scale with the feature
        config.rbmax_p = config.rbmax;
        config.rbmax_n = -config.rbmax;
        
        if(rmax_n == 0) rmax_n = -rowsum_p;  
        int iteration = 0;



        if(init)
        {
            for(uint i=0; i<vert; i++)
            {
                R[w][i] = feats(i, w);
                feats(i, w) = 0;
                if(R[w][i]>rmax_p || R[w][i]<rmax_n)
                {
                    candidate_set.push(i);
                    isCandidate[i] = true;
                }
            }
        }
        // cout<<"candidate_set.size(): "<<candidate_set.size()<<endl;
        while(candidate_set.size() > 0)
        {
            uint tempNode = candidate_set.front();
            candidate_set.pop();
            isCandidate[tempNode] = false;
            double old = R[w][tempNode];
            R[w][tempNode] = 0;
            feats(tempNode,w) += alpha*old;
            
            uint inSize = g.getInSize(tempNode);
            for(uint i=0; i<inSize; i++)
            {
                uint v = g.getInVert(tempNode, i);
                R[w][v] += (1-alpha) * old / Du[v] / Du[tempNode];
                if(!isCandidate[v])
                {
                    if(R[w][v] > rmax_p || R[w][v] < rmax_n)
                    {
                        candidate_set.push(v);
                        isCandidate[v] = true;
                    }
                }
            }
        }

        
        vector<bool>().swap(isCandidates[w]);
    }

}

}


// void Instantgnn::ppr_residue(Eigen::Ref<Eigen::MatrixXd>feats,int st,int ed, bool init,vector<queue<uint>>& candidate_sets,vector<vector<bool>>& isCandidates, string algorithm, bool reverse)
// {
//     // string algorithm = "instant";
//     // string algorithm = "speed_push";
    
//     int w;
//     for(int it=st;it<ed;it++)
//     {
//         if(init)
//             w = random_w[it];
//         else
//             w = update_w[it];

//         queue<uint> candidate_set = candidate_sets[w];
//         // queue<uint> candidate_set;
//         vector<bool> isCandidate = isCandidates[w];
//         vector<bool>().swap(isCandidates[w]);
//         queue<uint>().swap(candidate_sets[w]);
        
//         double rowsum_p=rowsum_pos[w];
//         double rowsum_n=rowsum_neg[w];
//         double current_L1;
//         if(init)
//             current_L1 = rowsum_p - rowsum_n;
//         else
//             current_L1 = R_sum_pos[w] - R_sum_neg[w];
//         double rmax_p=rowsum_p*rmax;
//         double rmax_n=rowsum_n*rmax;// to scale with the feature
        
//         config.rbmax_p = config.rbmax;
//         config.rbmax_n = -config.rbmax;
        
//         if(rmax_n == 0) rmax_n = -rowsum_p;  
//         int iteration = 0;


        
        
//         int number_epoch = 8;
//         uint i = 0;
//         double new_rmax = rmax;
//         rmax_p=rowsum_p*new_rmax;
//         rmax_n=rowsum_n*new_rmax;// to scale with the feature
//         double L1_thr = (rmax_p-rmax_n)*vert;
//         // cout<<" new_rmax:"<< new_rmax<<" L1_thr:"<<L1_thr<<endl;
//         // exit(0);
//         if(init)
//         {
//             for(uint i=0; i<vert; i++)
//             {
//                 R[w][i] = feats(i, w);
//                 feats(i, w) = 0;
//                 // if(R[w][i]>rmax_p || R[w][i]<rmax_n)
//                 // {
//                 //     candidate_set.push(i);
//                 //     isCandidate[i] = true;
//                 // }
//             }
//         }
//         // else{
//         //     for(uint i=0; i<vert; i++)
//         //     {
//         //         if(R[w][i]>rmax_p || R[w][i]<rmax_n)
//         //         {   
//         //             if(isCandidate[i]==false){
//         //                 candidate_set.push(i);
//         //                 isCandidate[i] = true;
//         //             }
//         //         }
//         //     }
//         // }
//         //// **The naive push method**
//         // for(int epoch =1; epoch<number_epoch+1; epoch++){
//         //     new_rmax = pow(rmax, double(epoch)/number_epoch);
            
//         //     rmax_p=rowsum_p*new_rmax;
//         //     rmax_n=rowsum_n*new_rmax;// to scale with the feature
//         //     L1_thr = (rmax_p-rmax_n)*vert;
//         //     // cout<<" new_rmax:"<< new_rmax<<" L1_thr:"<<L1_thr<<endl;
//         //     if(w==2)
//         //         cout<<"****current_L1:"<<current_L1<<" L1_thr"<< L1_thr<<" candidate_set.size():"<<candidate_set.size()<<endl;
//         //     while(current_L1>L1_thr&&candidate_set.size() > 0)
//         //     // while(candidate_set.size() > 0)
//         //     {   
                
//         //         uint tempNode = candidate_set.front();
//         //         candidate_set.pop();
//         //         isCandidate[tempNode] = false;
//         //         double old = R[w][tempNode];
//         //         R[w][tempNode] = 0;
//         //         feats(tempNode,w) += alpha*old;
//         //         i+=1;
//         //         if(i%10000==0&&w==2)
//         //             cout<<"current_L1:"<<current_L1<<" L1_thr"<< L1_thr<<" candidate_set.size():"<<candidate_set.size()<<" old:"<<old<<endl;
                
//         //         if(old>0)
//         //             current_L1-= alpha*old;
//         //         else
//         //             current_L1+= alpha*old;

//         //         uint inSize = g.getInSize(tempNode);
//         //         for(uint i=0; i<inSize; i++)
//         //         {
//         //             uint v = g.getInVert(tempNode, i);
//         //             R[w][v] += (1-alpha) * old / Du[v] / Du[tempNode];
//         //             if(!isCandidate[v])
//         //             {
//         //                 if(R[w][v] > rmax_p || R[w][v] < rmax_n)
//         //                 {
//         //                     candidate_set.push(v);
//         //                     isCandidate[v] = true;
//         //                 }
//         //             }
//         //         }
//         //     }
//         // }
//         //  **The end of naive**
        
//         // **The speed PPR method**
//         // const double time_start = getCurrentTime();
//         double one_minus_alpha = 1-alpha;
//         double real_r_sum = one_minus_alpha*current_L1;
//         uint number_of_pushes = 0;
//         uint previous_number_of_pushes = 0;
//         for(int epoch =0; epoch<number_epoch&&real_r_sum>L1_thr; ++epoch){
            
//             new_rmax = pow(rmax, (1.0+epoch)/number_epoch);
            
//             rmax_p=rowsum_p*new_rmax;
//             rmax_n=rowsum_n*new_rmax;// to scale with the feature
//             double l1_error_this_epoch = (rmax_p-rmax_n)*vert;
//             if(w==2)
//                     cout<<"****current real_r_sum:"<<real_r_sum<<" l1_error_this_epoch"<< l1_error_this_epoch<<endl;
//             uint32_t num_iter = log(2.0*real_r_sum / L1_thr) / log(1.0 / (1.0 - alpha));
//             // cout<<"num_iter: "<< num_iter<<endl;
//             for (unsigned int k = 0; k < num_iter && real_r_sum > l1_error_this_epoch; ++k) {
                
//                 // if(w==2)
//                 //     cout<<"****current real_r_sum:"<<real_r_sum<<" l1_error_this_epoch"<< l1_error_this_epoch<<endl;
//                 for(uint id=0; id<vert; ++id)
//                     {   
                        
//                         uint tempNode = id;
//                         uint inSize = g.getInSize(tempNode);
//                         // if(i%10000==0&&w==2)
//                         //     cout<<" R[w][tempNode]" <<R[w][tempNode]<<endl;
//                         if(R[w][tempNode] > rmax_p*Du[tempNode] || R[w][tempNode] < rmax_n*Du[tempNode]){
//                             double old = R[w][tempNode];
//                             R[w][tempNode] = 0;
//                             feats(tempNode,w) += alpha*old;
//                             i+=1;
//                             // if(i%10000==0&&w==2)
//                             //     cout<<"current_L1:"<<current_L1<<" L1_thr"<< L1_thr<<" old:"<<old<<endl;
                            
//                             if(old>0)
//                                 current_L1-= alpha*old;
//                             else
//                                 current_L1+= alpha*old;

//                             double increment = (1-alpha) * old / Du[tempNode];
//                             for(uint i=0; i<inSize; i++)
//                             {
//                                 uint v = g.getInVert(tempNode, i);
//                                 R[w][v] += increment/ Du[v];
//                             }
//                             number_of_pushes += inSize;
//                         }

//                         if ((number_of_pushes - previous_number_of_pushes >= edges)&&(w==2)) {
//                             previous_number_of_pushes = number_of_pushes;
//                             const size_t num_round = number_of_pushes / edges;
//                             real_r_sum = one_minus_alpha*current_L1;
//                             // const double time_used = getCurrentTime() - time_start;
//                             printf("#Iter:%s%lu\tr_sum:%.12f\t#Pushes:%zu\n",
//                                 (num_round < 10 ? "0" : ""), num_round, real_r_sum,
//                                 number_of_pushes);
//                         }

                        
//                     }
//                     real_r_sum = one_minus_alpha*current_L1;
//             }
//         }

        
 
//         // vector<bool>().swap(isCandidates[w]);
//         // queue<uint>().swap(candidate_sets[w]);
//     }


// }

// }