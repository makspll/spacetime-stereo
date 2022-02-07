from models.STSEarlyFusionTimeMatch import STSEarlyFusionTimeMatch
from models.STSEarlyFusionConcat import STSEarlyFusionConcat
from models.STSEarlyFusionConcat2 import STSEarlyFusionConcat2
from models.STSEarlyFusionConcat2Big import STSEarlyFusionConcat2Big
from models.STSLateFusion2 import STSLateFusion2
from models.LEAStereo import LEASTereoOrigMock, LEAStereo

import re 

def apply_state_dict_changes(s_dict,filters,replacings):
    new_state_dict = {}
    for k in s_dict:
        if k in filters:
            continue
        

        n_k = k
        for r in replacings:
            n_k = r(n_k)

        new_state_dict[n_k] = s_dict[k]
    filters.clear()
    replacings.clear()
    return new_state_dict

def convert_weights(state_dict,weights_source, weights_target):
    replacings = []
    filters = set()
    nets_altering_cost_volume = set([STSEarlyFusionConcat,STSEarlyFusionConcat2,STSEarlyFusionConcat2Big,STSEarlyFusionTimeMatch])
    nets_altering_disparity_outputs = set([STSEarlyFusionConcat2,STSEarlyFusionConcat2Big,STSEarlyFusionTimeMatch,STSLateFusion2])

    if (weights_source is LEASTereoOrigMock):
        print("===> converting from original LEAStereo weights")

        # for actual LEAStereo weights 
        replacings += [
            # feature cells
            lambda s: re.sub(r'(feature\.cells\..+\.)_ops\.0','\\1conv_prev_prev_to_zero',s), # 0 
            lambda s: re.sub(r'(feature\.cells\..+\.)_ops\.1','\\1skip_prev_to_zero',s), # 1
            lambda s: re.sub(r'(feature\.cells\..+\.)_ops\.2','\\1conv_prev_to_one',s), # 3
            lambda s: re.sub(r'(feature\.cells\..+\.)_ops\.3','\\1conv_zero_to_one',s), # 4
            lambda s: re.sub(r'(feature\.cells\..+\.)_ops\.4','\\1conv_prev_prev_to_two',s), # 5
            lambda s: re.sub(r'(feature\.cells\..+\.)_ops\.5','\\1conv_one_to_two',s), # 8

            # feature output
            lambda s: re.sub(r'(feature\.)last_3','\\1conv_out',s),

            # matching cells
            lambda s: re.sub(r'(matching\.cells\..+\.)_ops\.0','\\1conv_prev_prev_to_zero',s), # 0 
            lambda s: re.sub(r'(matching\.cells\..+\.)_ops\.1','\\1conv_prev_to_zero',s), # 1
            lambda s: re.sub(r'(matching\.cells\..+\.)_ops\.2','\\1conv_prev_to_one',s), # 3
            lambda s: re.sub(r'(matching\.cells\..+\.)_ops\.3','\\1conv_zero_to_one',s), # 4
            lambda s: re.sub(r'(matching\.cells\..+\.)_ops\.4','\\1conv_prev_to_two',s), # 6
            lambda s: re.sub(r'(matching\.cells\..+\.)_ops\.5','\\1conv_one_to_two',s), # 8
            
            # matching output
            lambda s: re.sub(r'(matching\.)last_3','\\1conv_out',s),

            # matching skips

            lambda s: re.sub(r'(matching\.)conv1','\\1skips.0',s),
            lambda s: re.sub(r'(matching\.)conv2','\\1skips.1',s),
        ]

        filters.update(["module.feature.last_6.conv.weight", "module.feature.last_6.bn.weight", "module.feature.last_6.bn.bias", "module.feature.last_6.bn.running_mean", "module.feature.last_6.bn.running_var", "module.feature.last_6.bn.num_batches_tracked", "module.feature.last_12.conv.weight", "module.feature.last_12.bn.weight", "module.feature.last_12.bn.bias", "module.feature.last_12.bn.running_mean", "module.feature.last_12.bn.running_var", "module.feature.last_12.bn.num_batches_tracked", "module.feature.last_24.conv.weight", "module.feature.last_24.bn.weight", "module.feature.last_24.bn.bias", "module.feature.last_24.bn.running_mean", "module.feature.last_24.bn.running_var", "module.feature.last_24.bn.num_batches_tracked", "module.matching.last_12.conv.weight", "module.matching.last_12.bn.weight", "module.matching.last_12.bn.bias", "module.matching.last_12.bn.running_mean", "module.matching.last_12.bn.running_var", "module.matching.last_12.bn.num_batches_tracked", "module.matching.last_24.conv.weight", "module.matching.last_24.bn.weight", "module.matching.last_24.bn.bias", "module.matching.last_24.bn.running_mean", "module.matching.last_24.bn.running_var", "module.matching.last_24.bn.num_batches_tracked"])
        state_dict = apply_state_dict_changes(state_dict,filters,replacings)
        if(weights_target in nets_altering_cost_volume):
            print("===> converting to " + "STSEarlyFusionConcat weights")
            filters.update(["module.matching.cells.0.pre_preprocess.conv.weight",
                "module.matching.cells.0.preprocess.conv.weight",
                "module.matching.cells.1.pre_preprocess.conv.weight",
                "module.matching.stem0.conv.weight",
                "module.matching.stem0.bn.weight",
                "module.matching.stem0.bn.bias",
                "module.matching.stem0.bn.running_mean",
                "module.matching.stem0.bn.running_var",
                "module.matching.stem1.conv.weight",
                "module.matching.stem1.bn.weight",
                "module.matching.stem1.bn.bias",
                "module.matching.stem1.bn.running_mean",
                "module.matching.stem1.bn.running_var"])

        # nets which output multiple disparities, or change matching network output 
        if weights_target in nets_altering_disparity_outputs: #not STSEarlyFusionConcat and not weights_target is LEAStereo):
            print("===> converting to " + str(weights_target) )
            filters.update([ "module.matching.conv_out.conv.weight",
                        "module.matching.conv_out.bn.weight",
                        "module.matching.conv_out.bn.bias",
                        "module.matching.conv_out.bn.running_mean",
                        "module.matching.conv_out.bn.running_var"])

    return apply_state_dict_changes(state_dict,filters,replacings)