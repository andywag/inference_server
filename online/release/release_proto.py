
import sys
sys.path.append("../model_proto")
sys.path.append("../public_api")

from project_proto import ProjectProto
from squad_proto import SquadProto
from ner_proto import NerProto
from gpt2_proto import GPT2Proto
from bart_proto import BartProto

import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser()   
    parser.add_argument("mode", type=str, default="run", choices=["run","test","fastapi"])
    parser.add_argument('--config',type=str,default='server')

    args = parser.parse_args()

    return args

models_map = {
    'squad' : SquadProto(),
    'bart': BartProto(),
    'ner' : NerProto(),
    'gpt2' : GPT2Proto()
}

def run(args):
    with open('config.yml') as fp:
        config = yaml.safe_load(fp)
        config = config[args.config]
        
        models = config['enabled']
        model_list = list()
        model_dict = dict()
        for model in models:
            model_list.append(models_map[model])
            model_dict[model] = models_map[model]
        project_proto = ProjectProto(name='base',models=model_list,models_dict=model_dict)
        project_proto.run_ipus(config)

if __name__ == '__main__':
    args = get_args()
    if args.mode == 'run':
        run(args)
        
    elif args.mode == 'test':
        project_proto.single_client_test(args.model, batch_size=args.batch_size, threads=args.threads)
    else:
        print("Mode not supported")