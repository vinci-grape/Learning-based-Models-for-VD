
import subprocess
import signal
import io
from typing import Union
import dot_util

_EDGE_LABEL_TYPES = ['ARGUMENT', 'AST', 'BINDS', 'CALL', 'CDG', 'CFG', 'CONDITION',
                    'CONTAINS', 'DOMINATE', 'EVAL_TYPE', 'PARAMETER_LINK',
                    'POST_DOMINATE', 'REACHING_DEF', 'RECEIVER', 'REF', 'SOURCE_FILE']

class JoernGraph:
    def __init__(self,nodes,edges):
        self.nodes = nodes
        self.edges = edges

    def __filter_edge(self,edges, edge_type):
        assert edge_type in _EDGE_LABEL_TYPES
        return list(filter(lambda x: x['label'] == edge_type, edges))

    def create_dot_graph(self,edge_type : Union[str,None] = None):
        edges = self.__filter_edge(self.edges,edge_type)
        nodes = []
        # remove some node that do not exist in edge
        appear_nodes = set([
            item
            for sub_tuple in [(e['inNode'],e['outNode']) for e in edges] for item in sub_tuple
        ])
        for n in self.nodes:
            if n['id'] in appear_nodes:
                nodes.append(n)

        graph = dot_util.DotGraphGenerator()
        for n in nodes:
            graph.node(n['id'], f"{n['_label']}  {n['name'] if 'name' in n.keys() else ''}  {n['code'] if 'code' in n.keys() else ''}")
        for e in edges:
            graph.edge(e['inNode'], e['outNode'])
        return graph

    def show_dot_graph(self,edge_type : Union[str,None] = None):
        g = self.create_dot_graph(edge_type)
        print(g.to_dot_file())
        g.open_in_browser()

from utils import read_json_file
nodes = read_json_file('processed/graph/tensorflow/8a47a39d9697969206d23a523c977238717e8727/tensorflow-core-kernels-mkl-mkl_qmatmul_op_test-cc/vul/after/2/2.nodes.json')
edges = read_json_file('processed/graph/tensorflow/8a47a39d9697969206d23a523c977238717e8727/tensorflow-core-kernels-mkl-mkl_qmatmul_op_test-cc/vul/after/2/2.edges.json')
g = JoernGraph(nodes,edges)
g.show_dot_graph('AST')