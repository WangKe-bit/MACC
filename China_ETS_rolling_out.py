import pandas as pd
import numpy as np
import os
import re
import time

df = pd.read_excel(r'..\data\ETSdataset.xlsx')
df['CO2_expand_per1'] = df['CO2_direct_expand'] * 0.01

designated_14_code = [2511,2614,2619,2621,3011,3041,3120,3211,3216,2211,2212,2221,4411,4420]
all_64_code = [1331,1340,1391,1440,1461,1469,1511,1512,1712,1713,2211,2212,2221,2511,2520,2611,2612,2613,2614,2619,2621,2622,2624,2651,2661,2664,2669,2770,2811,2812,2911,3011,3012,3021,3022,3024,3029,3032,3041,3052,3071,3073,3089,3110,3120,3130,3140,3150,3211,3212,3213,3216,3219,3261,3262,3340,3412,3423,3610,4411,4420,4430,4500,4610]
additional_50_code = [i for i in all_64_code if i not in designated_14_code]

#Create List from per1 to per99
per1_99 = []
for i in range(1,100):
    per1_99.append('per'+str(i))

#List all combinations: choose k elements from list L
def Combinations(L, k):
    n = len(L)
    result = []
    for i in range(n-k+1):
        if k > 1:
            newL = L[i+1:]
            Comb, _ = Combinations(newL, k - 1)
            for item in Comb:
                item.insert(0, L[i])
                result.append(item)
        else:
            result.append([L[i]])
    return result, len(result)

#Constructing a MACC for an industry combination using the firm's 99 MACs
def industry_combination_MACC(df_combination):
    global xy
    total_emission = df_combination['CO2_direct_expand'].sum()
    x = pd.DataFrame(np.tile(df_combination[['firmid','industry_code','CO2_expand_per1']],(99,1)),columns=['firmid','industry_code','CO2_expand_per1'])
    y = pd.DataFrame(np.matrix(df_combination[per1_99]).T.reshape(-1,1),columns=['price'])
    xy= pd.concat([y, x], axis=1)
    xy.sort_values(by='price', ignore_index=True, inplace = True)
    xy['CO2_cumulative'] = xy['CO2_expand_per1'].cumsum()
    xy['percent']=xy['CO2_cumulative']/total_emission * 100
    xy.index = range(1,len(xy)+1)
    return xy 

#The BAU case Stage I: include 13 designated industries in China's national ETS(4411)
def BAU_case_stage_I_cost_saving(df,indus_num):
    global keyindustrylist,df_industry_cost,df_group,industry_cost_list,result_cost_list
    keyindustrylist = [2511,2614,2619,2621,3011,3041,3120,3211,3216,2211,2212,2221,4411,4420]
    keyindustrylist.remove(4411)
    industrygroup = Combinations(keyindustrylist,indus_num)[0]
    industry_cost_list=[]
    result_cost_list=[]
    for i in range(len(industrygroup)):
        no_electricity = industrygroup[i][:]
        industrygroup[i].append(4411)
        df_group = pd.concat([df.loc[df['industry_code'].isin(no_electricity)],df_current_ETS])
        total_emission = df_group['CO2_direct_expand'].sum()
        total_product = df_group['totalproduct_expand_constant'].sum()*1000  
        industry_combination_MACC(df_group)
        #MACC of 4411
        current_ETS = xy[xy['industry_code'].isin([4411])]
        current_ETS.sort_values(by='price',ascending = True,inplace = True)
        current_ETS['CO2_cumulative']=np.cumsum(current_ETS.CO2_expand_per1)
        current_ETS['percent']=current_ETS['CO2_cumulative']/(current_ETS.CO2_cumulative.max()) * 100
        current_ETS.index = range(1,len(current_ETS)+1)

        every_percent_cost=[] 
        percent = {'1':5,'2':10,'3':15,'4':20}
        for goal in percent:
            percent_goal_list = []
            for x in range(1,percent[goal]+1):
                percent_goal_list.append('per'+str(x))
            
            #abatement cost under the ETS for 4411
            MACC_current_ETS = current_ETS[current_ETS['percent']<=(percent[goal]+0.01)]
            abatement_cost_4411_ETS = (MACC_current_ETS['price'] * MACC_current_ETS['CO2_expand_per1']).sum()*1000
            #abatement cost under the CCP for 4411
            abatement_cost_4411_CCP = (df_current_ETS[percent_goal_list].mean(axis=1)*percent[goal]*df_current_ETS['CO2_expand_per1']).sum()*1000
            
            #The abatement cost for 4420 is total industry product due to its small abatement potential
            if 4420 in no_electricity:
                Grid_abatement_cost = df_group[df_group['industry_code'].isin([4420])]['totalproduct_expand_constant'].sum()*1000
                no_Grid_no_electricity = no_electricity[:]
                no_Grid_no_electricity.remove(4420)
                df_group_no_Grid_no_electricity = df_group[df_group['industry_code'].isin(no_Grid_no_electricity)]
                abatement_cost_no_4411_CCP = Grid_abatement_cost + (df_group_no_Grid_no_electricity[percent_goal_list].mean(axis=1)*percent[goal]*df_group_no_Grid_no_electricity['CO2_expand_per1']).sum()*1000
            else:    
                df_group_no_electricity = df_group[df_group['industry_code'].isin(no_electricity)]
                abatement_cost_no_4411_CCP =  (df_group_no_electricity[percent_goal_list].mean(axis=1)*percent[goal]*df_group_no_electricity['CO2_expand_per1']).sum()*1000

            #abatement cost under the CCP for the industry combination (unit:CNY)
            abatement_cost_under_CCP = abatement_cost_no_4411_CCP + abatement_cost_4411_CCP 
            #abatement cost under the ETS for the industry combination (unit:CNY)
            MACC_under_ETS=xy[xy['percent']<=(percent[goal]+0.01)]
            abatement_cost_under_ETS = (MACC_under_ETS['price'] * MACC_under_ETS['CO2_expand_per1']).sum()*1000
            #abatement cost save (unit:CNY)
            cost_save = abatement_cost_under_CCP - abatement_cost_under_ETS
            #abatement cost saving (unit:%)
            cost_saving = cost_save/total_product*100
            #emissions reduction of the industry combination for a given reduction target
            abatement_amount = 0.01*percent[goal]*total_emission
            every_percent_cost.append([abatement_amount,abatement_cost_under_CCP,abatement_cost_under_ETS,cost_save,cost_saving])

        industrygroup_i = [str(a) for a in industrygroup[i]]
        industry_cost_list=['+'.join(industrygroup_i),len(df_group),total_emission,total_product]
        
        industry_cost_list.append(every_percent_cost)
        e=str(industry_cost_list)
        e=e.replace('[','').replace(']','')
        e=e.replace('nan','np.nan')
        elist=list(eval(e))
        result_cost_list.append(elist)
    
    industry_cost_header = ['industry_combination','total number of sample firms','total CO2 emission (ton)','total product (CNY)',\
                            '5% emissions reduction (ton)','5% abatement cost under CCP (CNY)','5% abatement cost under ETS (CNY)','5% cost save (CNY)','5% cost saving (%)',\
                            '10% emissions reduction (ton)','10% abatement cost under CCP (CNY)','10% abatement cost under ETS (CNY)','10% cost save (CNY)','10% cost saving (%)',\
                            '15% emissions reduction (ton)','15% abatement cost under CCP (CNY)','15% abatement cost under ETS (CNY)','15% cost save (CNY)','15% cost saving (%)',\
                            '20% emissions reduction (ton)','20% abatement cost under CCP (CNY)','20% abatement cost under ETS (CNY)','20% cost save (CNY)','20% cost saving (%)']

    df_industry_cost=pd.DataFrame(result_cost_list)
    df_industry_cost.columns = industry_cost_header    

#The BAU case Stage II: include 50 additional industries in China's national ETS(4411 + 13 designated industries)
def BAU_case_stage_II_cost_saving(df_group):
    global result,reduction_goal
    total_emission = df_group['CO2_direct_expand'].sum()
    total_product = df_group['totalproduct_expand_constant'].sum()*1000  
    industry_combination_MACC(df_group)
    percent = {'1':reduction_goal}
    for goal in percent:
        percent_goal_list = []
        for x in range(1,percent[goal]+1):
            percent_goal_list.append('per'+str(x))

        #The abatement cost for 4420 is total industry product due to its small abatement potential    
        Grid_abatement_cost = df_group[df_group['industry_code'].isin([4420])]['totalproduct_expand_constant'].sum()*1000
        no_Grid = indus_now_code[:]
        no_Grid.remove(4420)
        df_group_no_Grid = df_group[df_group['industry_code'].isin(no_Grid)]
        #abatement cost under the CCP for the industry combination (unit:CNY)
        abatement_cost_under_CCP = Grid_abatement_cost + (df_group_no_Grid[percent_goal_list].mean(axis=1)*percent[goal]*df_group_no_Grid['CO2_expand_per1']).sum()*1000  
        #abatement cost under the ETS for the industry combination (unit:CNY)
        MACC_under_ETS=xy[xy['percent']<=(percent[goal]+0.01)]
        abatement_cost_under_ETS = (MACC_under_ETS['price'] * MACC_under_ETS['CO2_expand_per1']).sum()*1000
        #abatement cost save (unit:CNY)
        cost_save = abatement_cost_under_CCP - abatement_cost_under_ETS
        #abatement cost saving (unit:%)
        cost_saving = cost_save/total_product*100
        #emissions reduction of the industry combination for a given reduction target
        abatement_amount = 0.01*percent[goal]*total_emission
        result.append([i,newcode,abatement_cost_under_CCP,abatement_cost_under_ETS,cost_save,total_product,cost_saving,abatement_amount])
            
#Find the optimal industry order of the 50 industries using the optimisation method  
def find_50_code():
    global result,final_result,newcode,indus_now_code,threshold,best_code,cost_saving_max
    result = []
    for newcode in additional_code:
        indus_now_code = ets_code[:]
        indus_now_code.append(newcode)
        df_group = df.loc[(df['industry_code'].isin(indus_now_code))&(df['year'] == 2011)&(df['CO2_adjust_finish'] >= threshold*1000)]
        BAU_case_stage_II_cost_saving(df_group)

    cost_save_percent = [x[6] for x in result]
    cost_saving_max = max(cost_save_percent)
    best_result = result[cost_save_percent.index(max(cost_save_percent))]
    best_code = best_result[1]
    final_result.append(best_result)
    additional_code.remove(best_code)
    ets_code.append(best_code)

#The Alternative case: include 63 industries in China's national ETS(4411)
def Alternative_case_cost_saving(df_group):
    global result,reduction_goal
    total_emission = df_group['CO2_direct_expand'].sum()
    total_product = df_group['totalproduct_expand_constant'].sum()*1000  
    industry_combination_MACC(df_group)
    percent = {'1':reduction_goal}
    for goal in percent:
        percent_goal_list = []
        for x in range(1,percent[goal]+1):
            percent_goal_list.append('per'+str(x))

        if 4420 in indus_now_code:
            Grid_abatement_cost = df_group[df_group['industry_code'].isin([4420])]['totalproduct_expand_constant'].sum()*1000
            no_Grid = indus_now_code[:]
            no_Grid.remove(4420)
            df_group_no_Grid = df_group[df_group['industry_code'].isin(no_Grid)]
            abatement_cost_under_CCP = Grid_abatement_cost + (df_group_no_Grid[percent_goal_list].mean(axis=1)*percent[goal]*df_group_no_Grid['CO2_expand_per1']).sum()*1000  
        else:
            no_Grid = indus_now_code[:]
            df_group_no_Grid = df_group[df_group['industry_code'].isin(no_Grid)]
            abatement_cost_under_CCP = (df_group_no_Grid[percent_goal_list].mean(axis=1)*percent[goal]*df_group_no_Grid['CO2_expand_per1']).sum()*1000
            
        MACC_under_ETS=xy[xy['percent']<=(percent[goal]+0.01)]
        abatement_cost_under_ETS = (MACC_under_ETS['price'] * MACC_under_ETS['CO2_expand_per1']).sum()*1000
        
        cost_save = abatement_cost_under_CCP - abatement_cost_under_ETS
        cost_saving = cost_save/total_product*100
        abatement_amount = 0.01*percent[goal]*total_emission
        result.append([i,newcode,abatement_cost_under_CCP,abatement_cost_under_ETS,cost_save,total_product,cost_saving,abatement_amount]) 

#Find the optimal industry order of the 63 industries using the optimisation method 
def find_63_code():
    global result,final_result,newcode,indus_now_code,threshold,best_code,cost_saving_max
    result = []
    for newcode in additional_code:
        indus_now_code = ets_code[:]
        indus_now_code.append(newcode)
        df_group = df.loc[(df['industry_code'].isin(indus_now_code))&(df['year'] == 2011)&(df['CO2_adjust_finish'] >= threshold*1000)]
        Alternative_case_cost_saving(df_group)

    cost_save_percent = [x[6] for x in result]
    cost_saving_max = max(cost_save_percent)
    best_result = result[cost_save_percent.index(max(cost_save_percent))]
    best_code = best_result[1]
    final_result.append(best_result)
    additional_code.remove(best_code)
    ets_code.append(best_code)



#The result of the BAU case Stage I under 12 scenarios
for threshold in [26,10,5]:
    df_sample = df[(df['year'] == 2011)&(df['CO2_adjust_finish'] >= threshold*1000)]
    df_current_ETS = df.loc[(df['industry_code'].isin([4411]))&(df['year'] == years)&(df['CO2_adjust_finish']>=threshold*1000)]
    
    start = time.process_time()
    with pd.ExcelWriter(r'..\\result\\The BAU case Stage I\\'+'The result of BAU case stage I '+str(threshold)+'kt entry threshold 2011.xlsx') as writer:
        for num in range(1,14):
            BAU_case_stage_I_cost_saving(df_sample,int(num))
            df_industry_cost.to_excel(writer, sheet_name = 'include '+str(int(num))+'designated industries',encoding='utf-8-sig')
    end = time.process_time()
    print('Run time: ',end-start)


#The result and process of the BAU case Stage II under 12 scenarios
for threshold in [5,10,26]:
    goal_list = [5,10,15,20]
    for reduction_goal in goal_list:
        additional_code = additional_50_code[:]
        ets_code = designated_14_code[:]
        final_result=[]
        column_list=['Round','industry code','abatement cost under CCP (CNY)','abatement cost under ETS (CNY)','cost save (CNY)','total product (CNY)','cost saving (%)','emissions reduction (ton)']
        df_process = pd.DataFrame(columns = column_list)
        
        for i in range(1,51):
            print('Round {} is being calculated'.format(i))
            find_50_code()
            print('Scenario({}kt&{}%): No.{} is {}, cost saving is {}%'.format(threshold,reduction_goal,i,best_code,cost_saving_max))
            df_pro = pd.DataFrame(result)
            df_pro.columns = column_list
            df_pro.index = range(1,len(df_pro)+1)
            df_process = pd.concat([df_process,df_pro])
            
        df_result = pd.DataFrame(final_result)
        df_result.columns = column_list
        df_result.index = range(1,len(df_result)+1)
        df_result.to_excel(r'..\\result\\The BAU case Stage II\\'+'The result of BAU case stage II '+str(threshold)+'kt entry threshold '+str(reduction_goal)+'% reduction target'+' 2011.xlsx')
        df_process.to_excel(r'..\\result\\The BAU case Stage II\\'+'The process of BAU case stage II '+str(threshold)+'kt entry threshold '+str(reduction_goal)+'% reduction target'+' 2011.xlsx')   


#The result and process of the Alternative case under 12 scenarios
for threshold in [5,10,26]:
    goal_list = [5,10,15,20]
    for reduction_goal in goal_list:
        additional_code = all_64_code[:]
        additional_code.remove(4411)
        ets_code = [4411]
        final_result=[]
        column_list=['Round','industry code','abatement cost under CCP (CNY)','abatement cost under ETS (CNY)','cost save (CNY)','total product (CNY)','cost saving (%)','emissions reduction (ton)']
        df_process = pd.DataFrame(columns = column_list)
        
        for i in range(1,64):
            print('Round {} is being calculated'.format(i))
            find_63_code()
            print('Scenario({}kt&{}%): No.{} is {}, cost saving is {}%'.format(threshold,reduction_goal,i,best_code,cost_saving_max))
            df_pro = pd.DataFrame(result)
            df_pro.columns = column_list
            df_pro.index = range(1,len(df_pro)+1)
            df_process = pd.concat([df_process,df_pro])
            
        df_result = pd.DataFrame(final_result)
        df_result.columns = column_list
        df_result.index = range(1,len(df_result)+1)
        df_result.to_excel(r'..\\result\\The Alternative case\\'+'The result of Alternative case '+str(threshold)+'kt entry threshold '+str(reduction_goal)+'% reduction target'+' 2011.xlsx')
        df_process.to_excel(r'..\\result\\The Alternative case\\'+'The process of Alternative case '+str(threshold)+'kt entry threshold '+str(reduction_goal)+'% reduction target'+' 2011.xlsx')   

#END




