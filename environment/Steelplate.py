import pandas as pd
import numpy as np
import functools
from environment.RL_SimComponent import Part

def import_steel_plate_schedule(filepath):
    imported_data = pd.read_csv(filepath)
    # print(imported_data)

    process_all = list()

    for i in range(1, 4):
        for process_name in list(imported_data['PROCESS{0}'.format(i)]):
            if process_name not in process_all:
                process_all.append(process_name)

    # print(process_all)

    process_list = ['CUT', 'BH', 'LH']

    machine_num = {
        'NC': {'Auto': 0, 'Manu': 1},
        'NG': {'Auto': 0, 'Manu': 1},
        'B0': {'Auto': 1, 'Manu': 0},
        'B1': {'Auto': 1, 'Manu': 0},
        'B2': {'Auto': 1, 'Manu': 0},
        'LH': {'Auto': 2, 'Manu': 6}
    }

    m_type_dict = {'NC': ['Manu{0}'.format(i + 1) for i in range(machine_num['NC']['Manu'])] +
                         ['Auto{0}'.format(i + 1) for i in range(machine_num['NC']['Auto'])],
                   'NG': ['Manu{0}'.format(i + 1) for i in range(machine_num['NG']['Manu'])] +
                         ['Auto{0}'.format(i + 1) for i in range(machine_num['NG']['Auto'])],
                   'B0': ['Manu{0}'.format(i + 1) for i in range(machine_num['B0']['Manu'])] +
                         ['Auto{0}'.format(i + 1) for i in range(machine_num['B0']['Auto'])],
                   'B1': ['Manu{0}'.format(i + 1) for i in range(machine_num['B1']['Manu'])] +
                         ['Auto{0}'.format(i + 1) for i in range(machine_num['B1']['Auto'])],
                   'B2': ['Manu{0}'.format(i + 1) for i in range(machine_num['B2']['Manu'])] +
                         ['Auto{0}'.format(i + 1) for i in range(machine_num['B2']['Auto'])],
                   'LH': ['Manu{0}'.format(i + 1) for i in range(machine_num['LH']['Manu'])] +
                         ['Auto{0}'.format(i + 1) for i in range(machine_num['LH']['Auto'])]
                   }

    # print(m_type_dict)

    # Distributions
    normal_dist = functools.partial(np.random.normal)

    ## Calculating process time of CUT PROCESS
    cut_mat_coeff = [abs(normal_dist(6, 5)) if list(imported_data['MATERIAL'])[i] == 'Mild Steel'
                     else abs(normal_dist(10, 5)) for i in range(len(imported_data))]
    imported_data['CUT_MAT_COEFF'] = cut_mat_coeff
    imported_data['CUT_THICKNESS_COEFF'] = imported_data['THICKNESS'] * 10 + imported_data['THICKNESS'] * imported_data[
        'THICKNESS']
    imported_data['CUT_LENGTH_WIDTH_COEFF'] = imported_data['LENGTH'] * imported_data['WIDTH'] * 0.2

    cut_form_coeff = list()
    for i in range(len(imported_data)):
        if list(imported_data['FORMINGLEVEL'])[i] == 1:
            coeff = abs(normal_dist(6, 5))
        elif list(imported_data['FORMINGLEVEL'])[i] == 2:
            coeff = abs(normal_dist(8, 5))
        else:
            coeff = abs(normal_dist(10, 5))
        cut_form_coeff.append(coeff)

    imported_data['CUT_FORM_COEFF'] = cut_form_coeff

    # CUT total working time (minutes)
    imported_data['CUT_WORKING_TIME'] = imported_data['CUT_MAT_COEFF'] + imported_data['CUT_THICKNESS_COEFF'] \
                                        + imported_data['CUT_LENGTH_WIDTH_COEFF'] + imported_data['CUT_FORM_COEFF']

    #print('average of CUT working time :  ', np.average(imported_data['CUT_WORKING_TIME']))

    ## Calculating process time of BH PROCESS
    bh_mat_coeff = [abs(normal_dist(8, 5)) if list(imported_data['MATERIAL'])[i] == 'Mild Steel'
                    else abs(normal_dist(12, 5)) for i in range(len(imported_data))]
    imported_data['BH_MAT_COEFF'] = bh_mat_coeff
    imported_data['BH_THICKNESS_COEFF'] = imported_data['THICKNESS'] * 5
    imported_data['BH_LENGTH_WIDTH_COEFF'] = imported_data['LENGTH'] * imported_data['WIDTH'] * 0.2

    bh_form_coeff = list()
    for i in range(len(imported_data)):
        if list(imported_data['FORMINGLEVEL'])[i] == 1:
            coeff = abs(normal_dist(8, 5))
        elif list(imported_data['FORMINGLEVEL'])[i] == 2:
            coeff = abs(normal_dist(10, 5))
        else:
            coeff = abs(normal_dist(12, 5))
        bh_form_coeff.append(coeff)

    imported_data['BH_FORM_COEFF'] = bh_form_coeff

    # BH total working time (minutes)
    imported_data['BH_WORKING_TIME'] = imported_data['BH_MAT_COEFF'] + imported_data['BH_THICKNESS_COEFF'] \
                                       + imported_data['BH_LENGTH_WIDTH_COEFF'] + imported_data['BH_FORM_COEFF']

    # print(imported_data['BH_WORKING_TIME'])
    #print('average of BH Working time :  ', np.average(imported_data['BH_WORKING_TIME']))

    ## Calculating process time of LH PROCESS
    lh_mat_coeff = [abs(normal_dist(60, 20)) if list(imported_data['MATERIAL'])[i] == 'Mild Steel'
                    else abs(normal_dist(100, 20)) for i in range(len(imported_data))]
    imported_data['LH_MAT_COEFF'] = lh_mat_coeff
    imported_data['LH_THICKNESS_COEFF'] = imported_data['THICKNESS'] * 10 + imported_data['THICKNESS'] * imported_data[
        'THICKNESS']
    imported_data['LH_LENGTH_WIDTH_COEFF'] = imported_data['LENGTH'] * imported_data['WIDTH'] * 0.6

    lh_form_coeff = list()
    for i in range(len(imported_data)):
        if list(imported_data['FORMINGLEVEL'])[i] == 1:
            coeff = abs(normal_dist(50, 20))
        elif list(imported_data['FORMINGLEVEL'])[i] == 2:
            coeff = abs(normal_dist(60, 20))
        else:
            coeff = abs(normal_dist(70, 20))
        lh_form_coeff.append(coeff)

    imported_data['LH_FORM_COEFF'] = lh_form_coeff

    # LH total working time (minutes)
    imported_data['LH_WORKING_TIME'] = imported_data['LH_MAT_COEFF'] + imported_data['LH_THICKNESS_COEFF'] \
                                       + imported_data['LH_LENGTH_WIDTH_COEFF'] + imported_data['LH_FORM_COEFF']

    # print(imported_data['LH_WORKING_TIME'])
    #print('average of LH working time :  ', np.average(imported_data['LH_WORKING_TIME']))

    ## Generating Multi-index dataframe
    columns = pd.MultiIndex.from_product(
        [[i for i in range(len(process_list) + 1)], ['start_time', 'process_time', 'process']])
    index = list(imported_data['WO_NO'])
    # print(index)
    df = pd.DataFrame(columns=columns, index=index)

    start_time = list(imported_data['SOURCE_IN'])
    # print(start_time)
    process_dict = dict()
    process_dict[0] = list(imported_data['PROCESS1'])
    process_dict[1] = list(imported_data['PROCESS2'])
    process_dict[2] = list(imported_data['PROCESS3'])
    process_time = dict()
    process_time[0] = list(imported_data['CUT_WORKING_TIME'])
    process_time[1] = list(imported_data['BH_WORKING_TIME'])
    process_time[2] = list(imported_data['LH_WORKING_TIME'])

    for i in range(len(process_list) + 1):
        if i == len(process_list):
            df[(i, 'start_time')] = None
            df[(i, 'process_time')] = None
            df[(i, 'process')] = 'Sink'
        else:
            df[(i, 'start_time')] = 0 if i != 0 else start_time
            df[(i, 'process_time')] = process_time[i]
            df[(i, 'process')] = process_dict[i]

    df = df.sort_values(by=[(0, 'start_time')], ascending=True)

    #print(df)
    return df, index, process_all, process_list, m_type_dict, machine_num

class SteelPlate(object):
    def __init__(self, df, index, process_list, m_type_dict):
        self.parts = list()

        parts_sent = 0
        while True:
            process_time_dict = dict()
            part_id, _data = df.index[parts_sent], df.iloc[parts_sent]
            tot_proc_time = 0
            for i in range(len(process_list)):
                tot_proc_time += df.loc[index[parts_sent], (i, 'process_time')]
            due_date = tot_proc_time * (1 + np.random.rand())
            part = Part(name=part_id, data=_data, process_time_dict=process_time_dict, due_date=due_date)


            step = 0
            while True:
                if part.data[(step, 'process')] != 'Sink':
                    process_time_dict[step] = dict()
                    for m_type in m_type_dict[part.data[(step, 'process')]]:
                        if m_type[:-1] == 'Auto':
                            process_time_dict[step][m_type] = part.data[(step, 'process_time')]
                        else:
                            process_time_dict[step][m_type] = part.data[(step, 'process_time')] * 1.2
                    step += 1
                else:
                    break

            part.process_time_dict = process_time_dict
            part.avg_proc_time = np.zeros(len(part.process_time_dict.keys()))
            part.min_proc_time = np.zeros(len(part.process_time_dict.keys()))
            part.max_proc_time = np.zeros(len(part.process_time_dict.keys()))

            for i in range(len(part.process_time_dict.keys())):
                avg_t = 0
                initial_key = list(part.process_time_dict[i].keys())[0]
                max_t = part.process_time_dict[i][initial_key]
                min_t = part.process_time_dict[i][initial_key]
                for key in part.process_time_dict[i].keys():
                    avg_t += part.process_time_dict[i][key]

                    if part.process_time_dict[i][key] > max_t:
                        max_t = part.process_time_dict[i][key]

                    if part.process_time_dict[i][key] < min_t:
                        min_t = part.process_time_dict[i][key]

                part.avg_proc_time[i] = avg_t / len(part.process_time_dict[i].keys())
                part.max_proc_time[i] = max_t
                part.min_proc_time[i] = min_t


            self.parts.append(part)
            # print(process_time_dict)

            parts_sent += 1
            # print(parts_sent)
            if parts_sent == len(df):
                #print('done')
                break

