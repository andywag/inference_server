
import os
import sys

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        project_path = os.getenv('TRITON_PROJECT_PROTO_PATH')
        sys.path.append(project_path)
        from release_proto import project_proto

        self.model_wrapper = project_proto.get_triton_interface(args['model_instance_name'])

        

    def execute(self, requests):
        return self.model_wrapper.execute(requests)

       

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
