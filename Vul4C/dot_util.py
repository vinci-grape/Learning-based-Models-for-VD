from dataclasses import dataclass
from typing import Union,Tuple
import os
import tempfile
import subprocess
# xdg-open directory
os.environ['PATH'] += ':' + '/home/root/.local/bin/'


def escape_string(str):
    s = ""
    for c in str:
        c = ord(c)
        if c   == ord('"') : s+= "&quot;"
        elif c == ord('<') : s+= "&lt;"
        elif c == ord('>') : s+= "&gt;"
        elif c == ord('&') : s+= "&amp;"
        elif c <= 0x9F and (c >= 0x7F or (c >> 5 == 0)): s += f"\\0{oct(c)[2:]}"
        else: s+= chr(c)
    return s

@dataclass
class DotEdge:
    src: int
    dst: int
    label: str

    def edge_to_dot(self):
        return f' "{self.src}" -> "{self.dst}" {"" if self.label is None else escape_string(self.label)}'

@dataclass
class DotNode:
    id:int
    label:Union[str,None]

    def node_to_dot(self):
        if self.label is None:
            return f'"{self.id}"'
        return f'"{self.id}" [label = <{escape_string(self.label)}>]'


class DotGraphGenerator:
    def __init__(self,name = "Test"):
        self.nodes : [DotNode]
        self.edges : [DotEdge]
        self.name = name
        self.nodes , self.edges = [],[]

    def node(self,id:int,label:Union[str,None] = None):
        self.nodes.append(DotNode(id, label))

    def edge(self,src:int,dst:int,label:Union[str,None] = None):
        self.edges.append(DotEdge(src,dst,label))

    def to_dot_file(self):
        s = f'digraph "{self.name}" {{  \n'
        for n in self.nodes:
            n:DotNode
            s += f"{n.node_to_dot()}\n"
        for e in self.edges:
            e:DotEdge
            s += f"{e.edge_to_dot()}\n"
        s += "\n}\n"
        return s

    def open_in_browser(self):
        """
            make sure you have installed xdg-open in system
        """
        graph = self.to_dot_file()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dot', delete=False) as dot_f:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as svg_f:
                dot_tmp_file_path = dot_f.name
                svg_tmp_file_path = svg_f.name
                dot_f.write(graph)
                dot_f.close()   # flush write content
                p = subprocess.run(f"dot -Tsvg {dot_tmp_file_path} -o {svg_tmp_file_path}",shell=True,check=True)
                print(dot_tmp_file_path)

                os.system(f"xdg-open {svg_tmp_file_path}")


