import argparse
import argcomplete
import numpy as np
from nntool.api import NNGraph
import sys
from PIL import Image,ImageDraw,ImageFont
import os
import pandas as pd 

CLASS_TYPE=['Background','Bottle','TinCan']
#######################################
### EXERCISE 3: ENABLE/DISABLE NE16 ###
#######################################
USE_NE16 = False    
#######################################


def main():

    im = Image.open('bottle_2_19.jpg').resize((320,240))
    
    
    im_array = ((np.array(im,dtype=np.uint8)[...,None].repeat(3,axis=2).astype(np.float32))-127)/128

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Path to DejaVu Sans
    font_size = 15
    font = ImageFont.truetype(font_path, font_size)
    at_model_dir = 'SSD_MODEL'

    G = NNGraph.load_graph('./models/SSD_tin_can_bottle.tflite', load_quantization=True)
    G.adjust_order()
    G.fusions("scaled_match_group")
    G.fusions('expression_matcher')

    G.quantize(
        None,
        schemes=['scaled'],
		#node_options={G[15].name:{'hwc': True,"use_ne16":False}},
		#{ G[i].name:{'hwc': True,"use_ne16":False} if(i==18) else {'hwc': True,"use_ne16":True}  for i in range(len(G))}
		
		#node_options={"
		#layer_option guardare come si fa 
        graph_options={
            #"use_ne16":True,
            "use_ne16_im2col":USE_NE16,
            "hwc": True,
			'allow_asymmetric_out':True
        },
        node_options={ G[-1].name:{'hwc': True,"use_ne16":False}}
    )

    G[0].allocate = True

    G.name = "ssd_Bottle_Tincan"

    setting_dictionary = {
        "tensor_directory": f"./tensors",
        "model_directory": f"./Execute_on_target",

        # Memory options
        "l1_size": 128000,
        ######################################
        ### EXERCISE 2: CHANGE THE L2 SIZE ###
        ######################################
        "l2_size": 1500000, 
        ######################################
        "l3_flash_device": 'AT_MEM_L3_DEFAULTFLASH',
        "l3_ram_device": 'AT_MEM_L3_DEFAULTRAM',
        "l3_ram_ext_managed": True,
        "privileged_l3_flash_device": "AT_MEM_L3_MRAMFLASH", #if True else "",
        "privileged_l3_flash_size": 1700000,

        # Autotiler Graph Options
        "graph_size_opt": 2, # Os for layers and xtor,dxtor,runner
        "graph_const_exec_from_flash": True,
        "graph_monitor_cycles": True,
        "graph_produce_node_names": True,
        "graph_produce_operinfos": True,
        "graph_trace_exec": False,
        "graph_async_fork":False,
    }
   
    output=G.execute_on_target(
        directory = './Execute_on_target',
        cmake = True,
        platform = 'board',
        finput_tensors = [im_array],
        write_out_to_file = True,
        output_tensors = False,
        at_log = True,
        at_loglevel = 2,
        settings = setting_dictionary,
        print_output = True, 
        performance=True
    )   

    performance = output.performance

    performance = {'Name':[per[0] for per in performance],'Cycles':[per[1] for per in performance],'MACs':[per[2] for per in performance],'MACs/Cycle':[per[3] for per in performance]} 
    
    pd.DataFrame(performance).to_json('Performance_Breakout.json' if not USE_NE16 else 'Performance_Breakout_NE16.json')

    boxes = np.fromfile('./Execute_on_target/Output_1.bin',dtype=np.int16).reshape((-1,4)).astype(np.float32)*6.103515625e-05
    classes = np.fromfile('./Execute_on_target/Output_2.bin',dtype=np.int8)
    scores = np.fromfile('./Execute_on_target/Output_3.bin',dtype=np.int8)

    drawn = ImageDraw.Draw(im)

    for box,classe,score in zip(boxes,classes,scores):
        print(box,classe,score)
        if score>54:
            drawn.rectangle((box[1]*320,box[0]*240,box[3]*320,box[2]*240),width=4)
            drawn.text((box[1]*320,box[0]*240-20), CLASS_TYPE[classe], font=font)

    im.show()
    im.save('./out_pred.jpg')


if __name__ == '__main__':
    main()
