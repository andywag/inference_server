
import sys
sys.path.append("../model_proto")
sys.path.append("../public_api")

from project_proto import ProjectProto
from squad_proto import SquadProto
from ner_proto import NerProto
from gpt2_proto import GPT2Proto
from bart_proto import BartProto


import argparse

def get_args():
    parser = argparse.ArgumentParser()   
    parser.add_argument("mode", type=str, default="run", choices=["run","test","fastapi"])
    parser.add_argument('--model', type=str, default="squad", choices=["squad","ner","gpt2"])
    parser.add_argument('--threads',type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()

    return args


project_proto = ProjectProto(
    name="base_models",
    #models=[NerProto(), GPT2Proto()]
    models=[BartProto(), NerProto(), GPT2Proto()]

)


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'run':
        project_proto.create_triton_structure("gen_models")
        project_proto.run_ipus()
    elif args.mode == 'test':
        project_proto.single_client_test(args.model, batch_size=args.batch_size, threads=args.threads)
    else:
        print("Mode not supported")