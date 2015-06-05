# coding: utf-8 

"""
======================================================================
Learning and Visualizing the BMS sensor-time-weather data structure
======================================================================
This example employs several unsupervised learning techniques to extract
the energy data structure from variations in Building Automation System (BAS) 
and historial weather data.
The fundermental timelet for analysis are 15 min, referred to as Q. 
** currently use H (Hour) as a fundermental timelet, need to change later **

The following analysis steps are designed and to be executed. 

Data Pre-processing
--------------------------
- Data Retrieval and Standardization
- Outlier Detection
- Interpolation 

Data Summarization
--------------------------
- Data Transformation
- Sensor Clustering

Model Discovery Bayesian Network
--------------------------
- Automatic State Classification
- Structure Discovery and Analysis
"""
#print(__doc__)
# Author: Deokwooo Jung deokwoo.jung@gmail.compile
##################################################################
# General Moduels
from __future__ import division # To forace float point division
from data_summerization import *
##################################################################
# Interactive mode for plotting
plt.ion() 

##################################################################
# Processing Configuraiton Settings
##################################################################
# Analysis buildings set
# Main building x where x is 1-16
# Conference bldg 
# Machine Room
# All Power Measurements

IS_USING_SAVED_DICT=-1
print 'Extract a common time range...'

##################################################################
# List buildings and substation names

PRE_BN_STAGE=0
if PRE_BN_STAGE==0:
    bldg_key_set=[]
    print 'skip PRE_BN_STAGE....'
else:
    bldg_key_set=['GW1','GW2','VAK1','VAK2']


#########################################
# 1. Electricity Room and Machine Room - 'elec_machine_room_bldg'
#########################################
#########################################
# 2. Conference Building - 'conference_bldg'
#########################################
#########################################
# 3. Main Building - 'main_bldg_x'
#########################################

for bldg_key in bldg_key_set:
    print '###############################################################################'
    print '###############################################################################'
    print 'Processing '+ bldg_key+'.....'
    print '###############################################################################'
    print '###############################################################################'

    temp=subprocess.check_output('ls '+DATA_DIR+'*'+bldg_key+'*.bin', shell=True)
    input_files_temp =shlex.split(temp)
    # Get rid of duplicated files
    input_files_temp=list(set(input_files_temp))
    input_files=input_files_temp
    #input_files=['../gvalley/Binfiles/'+temp for temp in input_files_temp]
    IS_USING_SAVED_DICT=0
    print 'Extract a common time range...'
    # Analysis period
    ANS_START_T=dt.datetime(2013,4,1,0)
    ANS_END_T=dt.datetime(2014,4,1,0)
    # Interval of timelet, currently set to 1 Hour
    TIMELET_INV=dt.timedelta(minutes=15)
    print TIMELET_INV, 'time slot interval is set for this data set !!'
    print '-------------------------------------------------------------------'
    PROC_AVG=True
    PROC_DIFF=True
    ###############################################################################
    # This directly searches files from bin file name
    print '###############################################################################'
    print '#  Data Pre-Processing'
    print '###############################################################################'
    # define input_files  to be read
    if IS_USING_SAVED_DICT==0:
        ANS_START_T,ANS_END_T,input_file_to_be_included=\
        time_range_check(input_files,ANS_START_T,ANS_END_T,TIMELET_INV)
        print 'time range readjusted  to  (' ,ANS_START_T, ', ', ANS_END_T,')'
        start__dictproc_t=time.time()
        if IS_SAVING_INDIVIDUAL==True:
            data_dict=construct_data_dict_2(input_files,ANS_START_T,ANS_END_T,TIMELET_INV,binfilename='data_dict', IS_USING_PARALLEL=IS_USING_PARALLEL_OPT)
        else:
            data_dict,purge_list=construct_data_dict(input_file_to_be_included,ANS_START_T,ANS_END_T,TIMELET_INV,binfilename='data_dict',IS_USING_PARALLEL=IS_USING_PARALLEL_OPT)
        end__dictproc_t=time.time()
        print 'the time of construct data dict.bin is ', end__dictproc_t-start__dictproc_t, ' sec'
        print '--------------------------------------'                
    elif IS_USING_SAVED_DICT==1:
        print 'Loading data dictionary......'
        start__dictproc_t=time.time()         
        data_dict = mt.loadObjectBinaryFast('data_dict.bin')
        end__dictproc_t=time.time()
        print 'the time of loading data dict.bin is ', end__dictproc_t-start__dictproc_t, ' sec'
        print '--------------------------------------'
    else:
        print 'Skip data dict' 
    
    CHECK_DATA_FORMAT=0
    if CHECK_DATA_FORMAT==1:
        if IS_SAVING_INDIVIDUAL==True:
            list_of_wrong_data_format=verify_data_format_2(data_used,data_dict,time_slots)
        else:
            list_of_wrong_data_format=verify_data_format(data_used,data_dict,time_slots)
        if len(list_of_wrong_data_format)>0:
            print 'Measurement list below'    
            print '----------------------------------------'    
            print list_of_wrong_data_format
            raise NameError('Errors in data format')
    Data_Summarization=1
    if Data_Summarization==1:
        bldg_out=data_summerization(bldg_key,data_dict,PROC_AVG=True,PROC_DIFF=False)
    
RECON_BLDG_BIN_OUT=0
if RECON_BLDG_BIN_OUT==1:
    for bldg_key in ['GW1_','GW2_','VAK1_','VAK2_']:
        avgdata_dict=mt.loadObjectBinaryFast('./VTT/'+bldg_key+'avgdata_dict.bin')
        diffdata_dict=mt.loadObjectBinaryFast('./VTT/'+bldg_key+'diffdata_dict.bin')
        data_dict=mt.loadObjectBinaryFast('./VTT/'+bldg_key+'data_dict.bin')
        cmd_str=remove_dot(bldg_key)+'out={\'data_dict\':data_dict}'
        exec(cmd_str)
        cmd_str=remove_dot(bldg_key)+'out.update({\'avgdata_dict\':avgdata_dict})'
        exec(cmd_str)
        cmd_str=remove_dot(bldg_key)+'out.update({\'diffdata_dict\':diffdata_dict})'
        exec(cmd_str)
        cmd_str=remove_dot(bldg_key)+'out.update({\'bldg_key\':remove_dot(bldg_key)})'
        exec(cmd_str)
        cmd_str='mt.saveObjectBinaryFast('+remove_dot(bldg_key)+'out'+',\''+PROC_OUT_DIR+remove_dot(bldg_key)+'out.bin\')'
        exec(cmd_str)
    

    
print '###############################################################################'
print '#  Model_Discovery'
print '###############################################################################'
bldg_key_set=['GW1','GW2','VAK1','VAK2']
Model_Discovery=1
if Model_Discovery==1:
    pwr_key='_POWER_';dict_dir='./VTT/'
    LOAD_BLDG_OBJ=0
    if LOAD_BLDG_OBJ==1:
        print 'not yet ready'
        bldg_=mt.loadObjectBinaryFast(PROC_OUT_DIR+'vtt_bldg_obj.bin')
    else:
        bldg_dict={}
        bldg_load_out = {}
        for bldg_load_key in  bldg_key_set:
            print 'Building for ',bldg_load_key, '....'
            try:
                bldg_tag='vtt_'+bldg_load_key
                bldg_load_out=mt.loadObjectBinaryFast(dict_dir+bldg_load_key+'_out.bin')
            except:
                print 'not found, skip....'
                pass
            mt.saveObjectBinaryFast(bldg_load_out['data_dict'],dict_dir+'data_dict.bin')
            if 'avgdata_dict' in bldg_load_out.keys():
                mt.saveObjectBinaryFast(bldg_load_out['avgdata_dict'],dict_dir+'avgdata_dict.bin')
            if 'diffdata_dict' in bldg_load_out.keys():
                mt.saveObjectBinaryFast(bldg_load_out['diffdata_dict'],dict_dir+'diffdata_dict.bin')
            pname_key= pwr_key
            bldg_dict.update({bldg_tag:create_bldg_obj(dict_dir,bldg_tag,pname_key)})
            bldg_=obj(bldg_dict)
            #cmd_str='bldg_.'+bldg_tag+'.data_out=obj(bldg_load_out)'
            #exec(cmd_str)
            cmd_str='bldg_obj=bldg_.'+bldg_tag
            exec(cmd_str)
            anal_out={}
            if 'avgdata_dict' in bldg_load_out.keys():
                anal_out.update({'avg':bn_prob_analysis(bldg_obj,sig_tag_='avg')})
            if 'diffdata_dict' in bldg_load_out.keys():
                anal_out.update({'diff':bn_prob_analysis(bldg_obj,sig_tag_='diff')})
            cmd_str='bldg_.'+bldg_tag+'.anal_out=obj(anal_out)'            
            exec(cmd_str)
        mt.saveObjectBinaryFast(bldg_ ,PROC_OUT_DIR+'vtt_bldg_obj.bin')
        mt.saveObjectBinaryFast('LOAD_BLDG_OBJ' ,PROC_OUT_DIR+'vtt_bldg_obj_is_done.txt')


def convert_vtt_name(id_labels):
    if isinstance(id_labels,list)==False:
        id_labels=[id_labels]
    out_name=[key_label_ for key_label_ in id_labels ]
    return out_name
bldg_.convert_name=convert_vtt_name
#######################################################################################
# Analysis For GSBC
#######################################################################################
# Analysis of BN network result
BN_ANAL=0
if BN_ANAL==1:
    # Plotting individual LHs
    PLOTTING_LH=0
    if PLOTTING_LH==1:
        plotting_bldg_lh(bldg_,attr_class='sensor',num_picks=30)
        plotting_bldg_lh(bldg_,attr_class='time',num_picks=30)
        plotting_bldg_lh(bldg_,attr_class='weather',num_picks=30)
    
    PLOTTING_BN=1
    if PLOTTING_BN==1:
        plotting_bldg_bn(bldg_)

More_BN_ANAL=0
if More_BN_ANAL==1:   
    #######################################################################################
    # Analysis For GSBC
    #######################################################################################
    #bldg_obj=bldg_.GSBC_main_bldg_power_machine_room
    bldg_obj=bldg_.GSBC_main_bldg_power_machine_room
    bldg_.GSBC_main_bldg_power_machine_room.anal_out=bn_prob_analysis(bldg_obj,sig_tag_='avg')
    bldg_obj=bldg_.GSBC_main_bldg_1
    bldg_.GSBC_main_bldg_1.anal_out=bn_prob_analysis(bldg_obj,sig_tag_='avg')
    
    import pdb;pdb.set_trace()
    #--------------------------------------------------------------------------
    # Analysis Display
    #--------------------------------------------------------------------------
    # Data set 1 - GSBC_main_bldg_power_machine_room
    p_name_sets_1=bldg_.GSBC_main_bldg_power_machine_room.anal_out.__dict__.keys()
    bn_out_sets_1=bldg_.GSBC_main_bldg_power_machine_room.anal_out.__dict__
    # Data set 2 - GSBC_main_bldg_1
    p_name_sets_2=bldg_.GSBC_main_bldg_1.anal_out.__dict__.keys()
    bn_out_sets_2=bldg_.GSBC_main_bldg_1.anal_out.__dict__
    
    # Data set 2 Analysis
    print 'List power meters for analysis'
    print '------------------------------------'
    pprint.pprint(np.array([p_name_sets_1,convert_gsbc_name(p_name_sets_1)]).T)
    print '------------------------------------'
    p_name=p_name_sets_1[3]
    bn_out=bn_out_sets_1[p_name]
       
    fig_name='BN for Sensors '+convert_gsbc_name(p_name)[0]
    fig=figure(fig_name,figsize=(30.0,30.0))
    col_name=[str(np.array([[lab1],[remove_dot(lab2)]])) \
    for lab1,lab2 in zip(bn_out.s_labels, convert_gsbc_name(bn_out.s_labels))]
    rbn.nx_plot(bn_out.s_hc,col_name,graph_layout='spring',node_text_size=15)
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    fig_name='BN for Time '+convert_gsbc_name(p_name)[0]
    fig=figure(fig_name,figsize=(30.0,30.0))
    rbn.nx_plot(bn_out.t_hc,convert_gsbc_name(bn_out.t_labels),graph_layout='spring',node_text_size=12)
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')    
    
    fig_name='BN for Weather '+convert_gsbc_name(p_name)[0]
    fig=figure(fig_name,figsize=(30.0,30.0))
    rbn.nx_plot(bn_out.w_hc,convert_gsbc_name(bn_out.w_labels),graph_layout='spring',node_text_size=12)
    png_name=fig_name+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    fig_name='BN for Sensor-Time-Weather '+convert_gsbc_name(p_name)[0]
    fig=figure(fig_name,figsize=(30.0,30.0))
    rbn.nx_plot(bn_out.all_hc,convert_gsbc_name(bn_out.all_labels),graph_layout='spring',node_text_size=20)
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')    
    
    fig_name='BN PEAK LH Analysis for Sensor-Time-Weather '+convert_gsbc_name(p_name)[0]
    fig=figure(fig_name, figsize=(30.0,30.0))
    subplot(2,1,1)
    plot(bn_out.all_cause_symbol_xtick,bn_out.high_peak_prob,'-^')
    plot(bn_out.all_cause_symbol_xtick,bn_out.low_peak_prob,'-v')
    plt.ylabel('Likelihood',fontsize='large')
    plt.xticks(bn_out.all_cause_symbol_xtick,bn_out.all_cause_symbol_xlabel,rotation=270, fontsize=10)
    plt.tick_params(labelsize='large')
    plt.legend(('High Peak', 'Low Peak'),loc='center right')
    plt.tick_params(labelsize='large')
    plt.grid();plt.ylim([-0.05,1.05])
    plt.title('Likelihood of '+ str(remove_dot(convert_gsbc_name(p_name)))+\
    ' given '+'\n'+str(remove_dot(convert_gsbc_name(bn_out.all_cause_label))))
    png_name=fig_name+'_'+str(uuid.uuid4().get_hex().upper()[0:2])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    
    # Compare with the raw data
    #-------------------------------------------    
    start_t=datetime.datetime(2013, 8, 9, 0, 0, 0)
    end_t=datetime.datetime(2013, 8, 13, 0, 0, 0)
    data_x=get_data_set([label_[2:] for label_ in bn_out.all_cause_label]+[p_name[2:]],start_t,end_t)
    png_namex=plot_data_x(data_x,stype='raw',smark='-^')
    png_namex=plot_data_x(data_x,stype='diff',smark='-^')
    
    name_list_out=[[p_name]+bn_out.all_cause_label,convert_gsbc_name([p_name]+bn_out.all_cause_label)]
    pprint.pprint(np.array(name_list_out).T)
    pprint.pprint(name_list_out)
    
    
    
    start_t=datetime.datetime(2013, 7, 1, 0, 0, 0)
    end_t=datetime.datetime(2013, 12, 31, 0, 0, 0)
    data_x=get_data_set([label_[2:] for label_ in bn_out.s_labels],start_t,end_t)
    png_namex=plot_data_x(data_x,stype='raw',smark='-^',fontsize='small',xpos=0.00)
    png_namex=plot_data_x(data_x,stype='diff',smark='-^')

    
    """
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    print '----------------------------------------'
    print 'Likelihoods '
    print '----------------------------------------'
    print cause_label+['Low Peak','High Peak']
    print '----------------------------------------'
    print np.vstack((np.int0(peak_state).T,np.int0(100*lowpeak_prob).T,np.int0(100*peak_prob).T)).T
    print '----------------------------------------'
    s_val_set=set(peak_state[:,0])
    m_val_set=set(peak_state[:,1])
    Z_peak=np.ones((len(s_val_set),len(m_val_set)))*np.inf
    for i,s_val in enumerate(s_val_set):
        for j,m_val in enumerate(m_val_set):
            idx=np.nonzero((peak_state[:,0]==s_val)&(peak_state[:,1]==m_val))[0][0]
            Z_peak[i,j]=peak_prob[idx]
            
    s_val_set=set(lowpeak_state[:,0])
    m_val_set=set(lowpeak_state[:,1])
    Z_lowpeak=np.ones((len(s_val_set),len(m_val_set)))*np.inf
    for i,s_val in enumerate(s_val_set):
        for j,m_val in enumerate(m_val_set):
            idx=np.nonzero((lowpeak_state[:,0]==s_val)&(lowpeak_state[:,1]==m_val))[0][0]
            Z_lowpeak[i,j]=lowpeak_prob[idx]
            
    Z_lowpeak=lowpeak_prob.reshape((len(s_val_set),len(m_val_set)))
    Z_peak=peak_prob.reshape((len(s_val_set),len(m_val_set)))
    fig1=figure()
    im = plt.imshow(Z_peak, cmap='hot',vmin=0, vmax=1,aspect='auto')
    plt.colorbar(im, orientation='horizontal')
    plt.xticks(monthDict.keys(),monthDict.values(),fontsize='large')
    plt.yticks(range(len(s_val_set)),list(s_val_set),fontsize='large')
    plt.xlabel(cause_label[1],fontsize='large')
    plt.ylabel(cause_label[0],fontsize='large')
    plt.title('Likelihood of High-Peak')
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig1.savefig(fig_dir+png_name+'.png', bbox_inches='tight')
    
    fig2=figure()
    im = plt.imshow(Z_lowpeak, cmap='hot',vmin=0, vmax=1,aspect='auto')
    plt.colorbar(im, orientation='horizontal')
    plt.xticks(monthDict.keys(),monthDict.values(),fontsize='large')
    plt.yticks(range(len(s_val_set)),list(s_val_set),fontsize='large')
    plt.xlabel(cause_label[1],fontsize='large')
    plt.ylabel(cause_label[0],fontsize='large')
    plt.title('Likelihood of Low-Peak')
    png_name=str(uuid.uuid4().get_hex().upper()[0:6])
    fig2.savefig(fig_dir+png_name+'.png', bbox_inches='tight')    
    
    """

print '**************************** End of Program ****************************'



