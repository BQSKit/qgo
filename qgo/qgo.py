import numpy as np
import time
import sys
import os

import qiskit as qk
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram, plot_gate_map, plot_circuit_layout
from qiskit import Aer, IBMQ, execute
from qiskit import QuantumCircuit, transpile
from datetime import date, datetime
from qiskit.transpiler.passes import StochasticSwap
from qiskit.transpiler.passes import Optimize1qGates
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import CHGate, U2Gate, CXGate
from qiskit.converters import circuit_to_dag
from qiskit.converters import dag_to_circuit
from qiskit.quantum_info.analysis import hellinger_fidelity
from qiskit.quantum_info import random_statevector, state_fidelity

from pytket.qiskit import qiskit_to_tk
from pytket.passes import PauliSimp
from pytket.backends.ibm import AerStateBackend
from pytket.qiskit import tk_to_qiskit
from pytket.routing import Architecture, route, LinePlacement
from pytket.device import Device
from pytket.circuit import Circuit
from pytket.predicates import CompilationUnit
from pytket.qasm import circuit_to_qasm, circuit_from_qasm
from pytket.passes import BasePass,CXMappingPass, RebaseIBM

import networkx as nx
import matplotlib.pyplot as plt

import search_compiler as sc
import qsearch
from qsearch import unitaries, advanced_unitaries, post_processing, leap_compiler, multistart_solver, parallelizer, utils, reoptimizing_compiler
#from qfast import *

def CouplingMap2DGrid(n_row, n_col):
    coupling_map = []
    for i in range(n_row):
        for j in range(n_col):
            idx = j + i*n_col
            if j != n_col - 1:
                coupling_map.append([idx, idx + 1])
            if i != n_row - 1:
                coupling_map.append([idx, idx + n_col])
    return coupling_map

def CouplingMapAllToAll(n_qubits):
    coupling_map = []
    for i in range(n_qubits):
        for j in range(i, n_qubits):
            coupling_map.append([i, j])
    return coupling_map

def num_device_qubits(coupling_map):
    nodes = set()
    for x,y in coupling_map:
        nodes.update([x,y])
    return len(nodes)
        
class RGate():
    def __init__(self, c = 0, t = 0, op = 0, isGate = True):
        self.c = c
        self.t = t
        self.op = op
        self.isGate = isGate
        
def IGate():
    return RGate(0,0,0,False)
        

class qgo():
    def __init__(self, circ, coupling_map, qubit_groups=None, reopt=True, reopt_single=True, verbosity=0, timeout=0, syn_tool='sc', b_size=3, cache=True, cache_threshold = 1e-8, sc_threshold = 1e-10, compose_threshold = 2e-8, name = 'new_circuit', block_details=True, all_to_all=False):
        print("RCircuit load circuit")
        self.qubit_array = []
        self.original_circuit = circ
        self.coupling_map = coupling_map
        self.qubit_groups = qubit_groups
        self.num_q = num_device_qubits(coupling_map)
        self.new_circuit = QuantumCircuit(self.num_q, self.num_q)
        self.b_size = b_size
        self.cache = cache
        self.cache_threshold = cache_threshold
        self.sc_threshold = sc_threshold
        self.compose_threshold = compose_threshold
        self.name = name
        self.block_details = block_details
        self.all_to_all = all_to_all
        self.syn_tool = syn_tool
        self.timeout = timeout
        self.verbosity = verbosity
        self.reopt_single = reopt_single
        
        # for output intermediate status, these variables will be changed during the process.
        self.qasm_blocks = []
        self.unitary_blocks = []
        self.new_qasm_blocks = []
        self.exe_time = []
        self.distance = []
        self.valid_blocks = []
        self.used_qubits = set()
        self.hit_rate = 0
        self.project_name = 'QGO_%s_q%s_b%s_%s_%s' % (self.name, self.num_q, self.b_size, str(self.sc_threshold), self.syn_tool)
        self.reopt = reopt
        
        if not os.path.exists(self.project_name):
            os.mkdir(self.project_name)
            
        self.tmp_new_circ = self.project_name + '/tmp_new_circ'
        if not os.path.exists(self.tmp_new_circ):
            os.mkdir(self.tmp_new_circ)
        
        for i in range(self.num_q):
            self.qubit_array.append([])
            
        for i in range(len(circ.data)):
            ins, qargs, cargs = circ.data[i]
            operation = RGate(qargs[0].index, qargs[-1].index, ins.qasm(), True)
            if operation.c == operation.t: #single-qubit operation
                self.qubit_array[operation.c].append(operation)
                self.used_qubits.add(operation.c)
            else: #two-qubit operation
                while(len(self.qubit_array[operation.c]) < len(self.qubit_array[operation.t])):
                    self.qubit_array[operation.c].append(IGate())
                while(len(self.qubit_array[operation.t]) < len(self.qubit_array[operation.c])):
                    self.qubit_array[operation.t].append(IGate())
                self.qubit_array[operation.c].append(operation)
                self.qubit_array[operation.t].append(operation)
                self.used_qubits.update([operation.c, operation.t])
                
        depth = self.circ_depth()
        for i in range(self.num_q):
            if i in self.used_qubits:
                while(len(self.qubit_array[i]) < depth):
                    self.qubit_array[i].append(IGate())
    
    def generate_qubit_groups(self):
        G = nx.Graph()
        qubit_groups = []
        for edge in self.coupling_map:
            G.add_edge(edge[0], edge[1])

        for s_idx in range(len(G.nodes)):
            for t_idx in range(len(G.nodes)):
                if s_idx in self.used_qubits and t_idx in self.used_qubits:
                    for path in nx.all_simple_paths(G, source=s_idx, target=t_idx, cutoff=self.b_size-1):
                        if len(path) >= 2:
                            qubit_groups.append(path)
        self.qubit_groups = qubit_groups

    def get_group_adjacency(self):
        group_adj = {}
        for group in self.qubit_groups:
            adj = []
            assigned_idx = 0
            idx_map = {}
            for q in group:
                idx_map[q] = assigned_idx
                assigned_idx += 1
            for edge in self.coupling_map:
                if edge[0] in group and edge[1] in group and [idx_map[edge[1]], idx_map[edge[0]]] not in adj:
                    adj.append([idx_map[edge[0]], idx_map[edge[1]]])
            group_adj[str(group)] = adj
        self.group_adj = group_adj
        
    def circ_depth(self):
        depth = 0
        for i in range(self.num_q):
            d = len(self.qubit_array[i])
            if d > depth:
                depth = d
        return depth
    
    def output_original_qasm(self):     
        qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        qasm_str += 'qreg q[%s];\n' % (self.num_q)
        depth = self.circ_depth()
        for j in range(depth):
            for i in range(self.num_q):
                operation =  self.qubit_array[i][j]
                if operation.isGate == True:
                    if operation.c == operation.t:
                        qasm_str += '%s q[%s];\n' % (operation.op, operation.t)
                    else:
                        if operation.c == i:
                            qasm_str += '%s q[%s],q[%s];\n' % (operation.op, operation.c, operation.t)
        return qasm_str
      
    def get_score(self, sched_depth, circ_depth, q_group):
        q_array = q_group[:]
        weight_one = 1
        weight_two = 1000
        weight_swap = 3 * weight_two
        score = 0
        qasm_str = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        qasm_str += 'qreg q[%s];\n' % (len(q_group))
        
        q_map = {}
        for i in range(len(q_array)):
            q_map[q_array[i]] = i
        
        finished_qubits = []

        for d in range(min(sched_depth), circ_depth):
            if len(q_array) == len(finished_qubits):
                break
            for q_idx in q_array:
                if q_idx not in self.used_qubits:
                    finished_qubits.append(q_idx)
                if d < sched_depth[q_idx]:
                    continue
                if q_idx in finished_qubits:
                    continue
                else:
                    operation = self.qubit_array[q_idx][d]
                    if operation.isGate == False:
                        sched_depth[q_idx] += 1
                    elif operation.isGate == True:
                        if operation.c == operation.t:
                            qasm_str += '%s q[%s];\n' % (operation.op, q_map[operation.t])
                            score += weight_one
                            sched_depth[q_idx] += 1
                        else:
                            if operation.c in q_array and operation.t in q_array and operation.c not in finished_qubits and operation.t not in finished_qubits:
                                sched_depth[q_idx] += 1
                                if operation.c == q_idx:
                                    qasm_str += '%s q[%s],q[%s];\n' % (operation.op, q_map[operation.c], q_map[operation.t])
                                    if operation.op == 'swap':
                                        score += weight_swap
                                    else:
                                        score += weight_two
                            else:
                                finished_qubits.append(q_idx)
                    
        return score, qasm_str, sched_depth
        
        
    def get_circ_block(self, sched_depth, circ_depth, qubit_groups):
        max_sched_depth = sched_depth[:]
        max_score = -1
        max_score_idx = -1
        max_score_qasm = ''
        for i in range(len(qubit_groups)):
            s_d = sched_depth[:]
            score, qasm, s_d = self.get_score(s_d, circ_depth, qubit_groups[i])
            if score > max_score:
                max_score = score
                max_score_idx = i
                max_score_qasm = qasm
                max_sched_depth = s_d
                q_mapping = qubit_groups[i]
            if score == max_score and len(qubit_groups[i]) < len(q_mapping):
                max_score = score
                max_score_idx = i
                max_score_qasm = qasm
                max_sched_depth = s_d
                q_mapping = qubit_groups[i]
        
        return max_score_qasm, max_sched_depth, q_mapping
    
    def circ_partition(self):
        circ_depth = self.circ_depth()
        sched_depth = []
        for i in range(self.num_q):
            if i not in self.used_qubits:
                sched_depth.append(circ_depth)
            else:
                sched_depth.append(0)
        
        qubit_groups = self.qubit_groups
        qasm_blocks = []
        qubit_mappings = []

        while(min(sched_depth) < circ_depth):
            if self.verbosity == 2:
                print("circuit partition: depth: %s/%s" % (sched_depth, circ_depth))
            qasm, sched_depth, q_mapping = self.get_circ_block(sched_depth, circ_depth, qubit_groups)
            qasm = qasm.replace("2pi", "2*pi").replace("3pi", "3*pi").replace("4pi", "4*pi").replace("5pi", "5*pi").replace("6pi", "6*pi").replace("7pi", "7*pi").replace("8pi", "8*pi").replace("9pi", "9*pi")

            qasm_blocks.append(qasm)
            qubit_mappings.append(q_mapping)

        self.qasm_blocks = qasm_blocks
        self.qubit_mappings = qubit_mappings
        print("Total #Blocks: %s" % (len(qasm_blocks)))
        return qasm_blocks, qubit_mappings

    def get_unitary(self):
        unitary_blocks = []
        backend = Aer.get_backend('unitary_simulator')
        
        for i in range(len(self.qasm_blocks)):
            circ = QuantumCircuit.from_qasm_str(self.qasm_blocks[i])
    
            job = execute(circ, backend)
            result = job.result()
            u_circ = result.get_unitary(circ)
            unitary_blocks.append(u_circ)

        self.unitary_blocks = unitary_blocks
        return unitary_blocks
    
    def get_new_unitary(self):
        new_unitary_blocks = []
        backend = Aer.get_backend('unitary_simulator')
        
        for i in range(len(self.new_qasm_blocks)):
            circ = QuantumCircuit.from_qasm_str(self.new_qasm_blocks[i])
            job = execute(circ, backend)
            result = job.result()
            u_circ = result.get_unitary(circ)
            new_unitary_blocks.append(u_circ)
            md = sc.utils.matrix_distance_squared(self.unitary_blocks[i], u_circ)
            self.distance.append(md)

        self.new_unitary_blocks = new_unitary_blocks
        return new_unitary_blocks

    
    def check_cache(self, base, prev_unitary_blocks, unitary):
        if self.cache == False:
            return -1
        
        for i in range(len(prev_unitary_blocks)):
            if len(prev_unitary_blocks[i]) == len(unitary):
                if abs(sc.utils.matrix_distance_squared(prev_unitary_blocks[i], unitary)) <= self.cache_threshold:
                    return i + base
        return -1
    
    def get_new_qasm_from_sc(self):
        num_hit = 0.0
        new_qasm_blocks = []
        exe_time = []
        if self.syn_tool == 'leap':
            print("Use LeapCompiler")
        else:
            print("Use regular SearchCompiler")
            
        project_name = self.project_name
        for i in range(len(self.unitary_blocks)):
            start_time = time.time()
            base = max(0, i - 4 * self.num_q)
            hit = -1
            if hit == -1:
                u_reversed = sc.utils.endian_reverse(self.unitary_blocks[i])
                if not os.path.exists('current_process'):
                    os.mkdir('current_process')
                np.savetxt('current_process/unitary.txt', self.unitary_blocks[i])
                f_cur =  open('current_process/status.txt', 'w')
                f_cur.write(str(self.group_adj[str(self.qubit_mappings[i])]))
                f_cur.write('\n')
                f_cur.write('circuit name: %s\n' % self.name)
                f_cur.write('block idx: %s\n' % i)
                f_cur.close()
                f_qasm = open('current_process/circ.qasm', 'w')
                f_qasm.write(self.qasm_blocks[i])
                f_qasm.close()
                
                
                tmp_project_name = 'tmp_%s_%s_%s_%s_%s' % (self.name, i, str(date.today()), datetime.now().hour, datetime.now().minute)
                
                # check if the circuit is generated previously.
                filename = self.tmp_new_circ + '/new_b_' + str(i) + '.qasm'
                if os.path.exists(filename):
                    new_circ_file = open(filename, 'r')
                    new_qasm = new_circ_file.read()
                    new_circ_file.close()
                    
                else:
                    if self.syn_tool == 'sc':
                        project = qsearch.Project(tmp_project_name)
                        project["gateset"] = qsearch.gatesets.QubitCNOTAdjacencyList(self.group_adj[str(self.qubit_mappings[i])])
                        project["threshold"] = self.sc_threshold
                        #project["timeout"] = self.timeout
                        project.add_compilation(self.name, u_reversed)
                        project["verbosity"] = self.verbosity
                        project.run()
                        #project.post_process(post_processing.BasicSingleQubitReduction_PostProcessor(), solver=multistart_solver.MultiStart_Solver(8))
                        new_qasm = project.assemble(self.name)
                        project.reset()
                    else:
                        project = qsearch.Project(tmp_project_name)
                        project["gateset"] = qsearch.gatesets.QubitCNOTAdjacencyList(self.group_adj[str(self.qubit_mappings[i])])
                        project["threshold"] = self.sc_threshold
                        project.add_compilation(self.name, u_reversed)
                        project["verbosity"] = self.verbosity
                        project["min_depth"] = 6
                        project["compiler_class"] = leap_compiler.LeapCompiler
                        project.run()
                        phase_1_qasm = project.assemble(self.name)

                        if self.reopt == True and phase_1_qasm.count('cx') > 0:
                            print("resynthesize")
                            reopt_proj_name = '%s_q%s_b%s_%s_%s_%s_reopt' % (self.name, self.num_q, self.b_size, str(date.today()), datetime.now().hour, datetime.now().minute)
                            with qsearch.Project(reopt_proj_name) as reoptimized:
                                for comp in project.compilations:
                                    target = project.get_target(comp)
                                    results = project.get_result(comp)
                                    circ = results['structure']
                                    cut_depths = results['cut_depths']
                                    best_pair = (circ, results['vector'])
                                    reoptimized.add_compilation(comp, target, cut_depths=cut_depths, best_pair=best_pair, depth=5)
                                reoptimized['compiler_class'] = reoptimizing_compiler.ReoptimizingCompiler
                                # Use the multistart solver for better (but slower) results, you may want to
                                # replace 2 with the number of processes you want to give to the optimizer (more processes ~= more accurate)
                                reoptimized["solver"] = multistart_solver.MultiStart_Solver(8)
                                # Multistart requires nested processes, so we use ProcessPoolExecutor
                                reoptimized["parallelizer"] = parallelizer.ProcessPoolParallelizer
                                reoptimized['verbosity'] = self.verbosity
                                reoptimized["threshold"] = self.sc_threshold
                                reoptimized["gateset"] = qsearch.gatesets.QubitCNOTAdjacencyList(self.group_adj[str(self.qubit_mappings[i])])
                                reoptimized.run()
                                tmp_qasm = reoptimized.assemble(self.name)
                                if tmp_qasm.count('cx') > 0 and self.reopt_single:
                                    reoptimized.post_process(post_processing.BasicSingleQubitReduction_PostProcessor(), solver=multistart_solver.MultiStart_Solver(8))
                                new_qasm = reoptimized.assemble(self.name)
                                reoptimized.reset()
                        else:
                            new_qasm = phase_1_qasm
                        
                        project.reset()
                    # write new qasm to tmp_new_circ folder
                    out_file = open(filename, 'w')
                    out_file.write(new_qasm)
                    out_file.close()
                    
            else:
                print("hit: %s" % (hit))
                num_hit += 1
                new_qasm = new_qasm_blocks[hit]
                
            syn_time = time.time() - start_time
            new_circ = qk.QuantumCircuit.from_qasm_str(new_qasm)
            new_circ = transpile(circuits=new_circ, basis_gates=['u3', 'cx'], optimization_level=3)
            new_qasm = str(new_circ.qasm()).replace("2pi", "2*pi").replace("3pi", "3*pi").replace("4pi", "4*pi").replace("5pi", "5*pi").replace("6pi", "6*pi").replace("7pi", "7*pi").replace("8pi", "8*pi").replace("9pi", "9*pi")

            new_qasm_blocks.append(new_qasm)
            exe_time.append(syn_time)
            print("progress: %s/%s, time: %.2f" % (i, len(self.unitary_blocks), syn_time))
            print("ori cx = %s, new cx = %s" % (self.qasm_blocks[i].count('cx') + 3 * self.qasm_blocks[i].count('swap'), new_qasm.count('cx')))
        
        self.new_qasm_blocks = new_qasm_blocks
        self.exe_time = exe_time
        
        self.hit_rate = num_hit / len(new_qasm_blocks)
        print("hit rate = %.3f" % (self.hit_rate))
        return new_qasm_blocks
    
    def get_new_qasm(self):
        self.get_new_qasm_from_sc()
        
    def compose_original_circuit(self):
        c_ori_circuit = QuantumCircuit(self.num_q, self.num_q)
        for i in range(len(self.qasm_blocks)):
            rhs = qk.QuantumCircuit.from_qasm_str(self.qasm_blocks[i])
            c_ori_circuit.compose(rhs,qubits=self.qubit_mappings[i], inplace=True)
        self.c_ori_circuit = c_ori_circuit
        return c_ori_circuit
    
    def compose_new_circuit(self):
        new_circuit = QuantumCircuit(self.num_q, self.num_q)

        for i in range(len(self.new_qasm_blocks)):
            ori_cx = self.qasm_blocks[i].count('cx') + 3 * self.qasm_blocks[i].count('swap')
            new_cx = self.new_qasm_blocks[i].count('cx')
            
            ori_circ = qk.QuantumCircuit.from_qasm_str(self.qasm_blocks[i])
            new_circ = qk.QuantumCircuit.from_qasm_str(self.new_qasm_blocks[i])
            
            new_circ = transpile(circuits=new_circ, basis_gates=['u3', 'cx'], optimization_level=3)
            
            ori_single = ori_circ.size() - ori_circ.num_nonlocal_gates()
            new_single = new_circ.size() - new_circ.num_nonlocal_gates()
            
            if new_cx < ori_cx:
                rhs = qk.QuantumCircuit.from_qasm_str(self.new_qasm_blocks[i])
                print("Use synthesized block for block-%s" % (i))
                self.valid_blocks.append(i)
            else:
                rhs = qk.QuantumCircuit.from_qasm_str(self.qasm_blocks[i])
            new_circuit.compose(rhs,qubits=self.qubit_mappings[i], inplace=True)
            
        self.new_circuit = new_circuit
        return self.new_circuit
    
    def post_process(self):
        new_circuit = self.new_circuit
        
        self.new_circuit = qk.transpile(circuits=new_circuit, basis_gates=['u3', 'cx'], optimization_level=3)
        
        #circ_tk = qiskit_to_tk(new_circuit)
        #tk_backend = AerStateBackend()
        #tk_backend.compile_circuit(circ_tk)
        #self.new_circuit = tk_to_qiskit(circ_tk)
        
        return self.new_circuit
    
    def get_state_fidelity(self, ori_circ, new_circ):
        num_qubits = ori_circ.num_qubits
        num_test = 10

        fid_all = 0
        
        for i in range(num_test):
            qc_ori = QuantumCircuit(num_qubits ,num_qubits)
            qc_new = QuantumCircuit(num_qubits ,num_qubits)
            init_state = random_statevector(2**num_qubits, np.random.randint(2000000))
            qc_ori.initialize(init_state.data, range(num_qubits))
            qc_new.initialize(init_state.data, range(num_qubits))

            qc_ori.compose(ori_circ, inplace=True)
            qc_new.compose(new_circ, inplace=True)

            backend = Aer.get_backend('statevector_simulator')
            ori_job = execute(qc_ori, backend)
            state_ori = ori_job.result().get_statevector(qc_ori)
            new_job = execute(qc_new, backend)
            state_new = new_job.result().get_statevector(qc_new)
            fid = qk.quantum_info.state_fidelity(state_ori, state_new)
            
            fid_all += fid

        return fid_all / num_test
        
            
    
    def run(self):
        str_time = ''
        
        all_time = time.time()
        print("Synthesize with %s" % (self.syn_tool))
        sys.stdout.flush()
        
        if self.qubit_groups == None:
            start_time = time.time()
            print("Generate qubit groups")
            self.generate_qubit_groups()
            print("generate_qubit_groups:\t%.2f" % (time.time() - start_time))
            str_time += "generate_qubit_groups:\t%.2f\n" % (time.time() - start_time)
            sys.stdout.flush()
        else:
            print("Costumized qubit groups are given.")
        
        start_time = time.time()
        self.circ_partition()
        print("circ_partition:\t%.2f" % (time.time() - start_time))
        str_time += "circ_partition:\t%.2f\n" % (time.time() - start_time)
        sys.stdout.flush()
        
        start_time = time.time()
        self.get_unitary()
        print("get_unitary:\t%.2f" % (time.time() - start_time))
        str_time += "get_unitary:\t%.2f\n" % (time.time() - start_time)
        sys.stdout.flush()
        
        start_time = time.time()
        self.get_group_adjacency()
        print("get_group_adjacency:\t%.2f" % (time.time() - start_time))
        str_time += "get_group_adjacency:\t%.2f\n" % (time.time() - start_time)
        sys.stdout.flush()
        
        start_time = time.time()
        self.get_new_qasm()
        print("get_new_qasm:\t%.2f" % (time.time() - start_time))
        str_time += "get_new_qasm:\t%.2f\n" % (time.time() - start_time)
        sys.stdout.flush()
        
        start_time = time.time()
        self.get_new_unitary()
        print("get_new_unitary:\t%.2f" % (time.time() - start_time))
        str_time += "get_new_unitary:\t%.2f\n" % (time.time() - start_time)
        sys.stdout.flush()
        
        start_time = time.time()
        self.compose_new_circuit()
        print("compose_new_circuit:\t%.2f" % (time.time() - start_time))
        str_time += "compose_new_circuit:\t%.2f\n" % (time.time() - start_time)
        sys.stdout.flush()
        
        start_time = time.time()
        self.post_process()
        print("post_process:\t%.2f" % (time.time() - start_time))
        str_time += "post_process:\t%.2f\n" % (time.time() - start_time)
        sys.stdout.flush()
        
        
        self.total_time = time.time() - all_time
        print("total time:\t%.2f" % (self.total_time))
        str_time += "total time:\t%.2f\n" % (self.total_time)
        sys.stdout.flush()
        
        self.str_time = str_time
        self.output_stats()
        print("Complete!")
        return self.new_circuit    
    
    def output_stats(self):
        distance = []
        details = self.block_details
        out_str = ''
        out_fidelity = ''
        out_block_details = ''
        
        if not os.path.exists(self.project_name):
            os.mkdir(self.project_name)
            
        filename_mapping = self.project_name + '/mapping.txt'
        np.savetxt(filename_mapping, self.qubit_mappings, fmt='%s')
        
        filename_valid = self.project_name + '/valid_blocks.txt'
        np.savetxt(filename_valid, self.valid_blocks, fmt='%s')
        
        
        
        # output original qasm blocks
        dirName = self.project_name + '/original_qasm_blocks'
        if not os.path.exists(dirName):
            os.mkdir(dirName)
    
        for i in range(len(self.qasm_blocks)):
            filename = dirName + '/b_' + str(i) + '.qasm'
            out_file = open(filename, 'w')
            out_file.write(self.qasm_blocks[i])
            out_file.close()
        
        # output new qasm blocks
        dirName = self.project_name + '/new_qasm_blocks'
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            
        for i in range(len(self.new_qasm_blocks)):
            filename = dirName + '/new_b_' + str(i) + '.qasm'
            out_file = open(filename, 'w')
            out_file.write(self.new_qasm_blocks[i])
            out_file.close()
        
        # output original unitaries
        dirName = self.project_name + '/original_unitaries'
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        for i in range(len(self.unitary_blocks)):
            filename = dirName + '/u_' + str(i)
            np.savetxt(filename, self.unitary_blocks[i])
        
        # overall stats
        circ = self.original_circuit
        new_circ = self.new_circuit
        
        qasm = new_circ.qasm().replace("2pi", "2*pi").replace("3pi", "3*pi").replace("4pi", "4*pi").replace("5pi", "5*pi").replace("6pi", "6*pi").replace("7pi", "7*pi").replace("8pi", "8*pi").replace("9pi", "9*pi")
    
        new_filename = self.project_name + "/new_circuit.qasm"
        f_new_qasm = open(new_filename, 'w')
        f_new_qasm.write(qasm)
        f_new_qasm.close()
        
        qasm = circ.qasm().replace("2pi", "2*pi").replace("3pi", "3*pi").replace("4pi", "4*pi").replace("5pi", "5*pi").replace("6pi", "6*pi").replace("7pi", "7*pi").replace("8pi", "8*pi").replace("9pi", "9*pi")
    
        ori_filename = self.project_name + "/ori_circuit.qasm"
        f_ori_qasm = open(ori_filename, 'w')
        f_ori_qasm.write(qasm)
        f_ori_qasm.close()
        
        if self.num_q <= 12:
            backend = Aer.get_backend('unitary_simulator')
            ori_job = execute(circ, backend)
            u_ori = ori_job.result().get_unitary(circ)
            new_job = execute(new_circ, backend)
            u_new = new_job.result().get_unitary(new_circ)
            dist = sc.utils.matrix_distance_squared(u_ori, u_new)
            
            fid_dist = 1 - self.get_state_fidelity(circ, new_circ)
            
        elif self.num_q <= 20:
            backend = Aer.get_backend('statevector_simulator')
            ori_job = execute(circ, backend)
            state_ori = ori_job.result().get_statevector(circ)
            new_job = execute(new_circ, backend)
            state_new = new_job.result().get_statevector(new_circ)
            fid_dist = qk.quantum_info.state_fidelity(state_ori, state_new)
            dist = -1
        else:
            fid_dist = -1
            dist = -1
        
        num_swap = circ.qasm().count('swap')
        new_num_swap = new_circ.qasm().count('swap')
        out_str_header = 'Benchmark\tQubits\t#ProgCNOTs\t#SwapCNOTs\t#CNOTs\t#SynCNOTs\t#1Q-Gates\tNew #1Q-Gates\tDepth\tNew Depth\tDistance\tInfidelity\tTime\t#Blocks\tCache\tHit Rate\tBlock Size\tSC_Threshold\tProjectName\n'
        out_str += '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f\t%s\t%s\t%.3f\t%s\t%s\t%s\n' % (self.name, \
                                                                                            len(self.used_qubits), \
                                                                                            circ.num_nonlocal_gates() - num_swap, \
                                                                                            3 * num_swap, \
                                                                                            circ.num_nonlocal_gates() + 2 * num_swap,
                                                                                            new_circ.num_nonlocal_gates() + 2 * new_num_swap, \
                                                                                            circ.size() - circ.num_nonlocal_gates(), \
                                                                                            new_circ.size() - new_circ.num_nonlocal_gates(), \
                                                                                            circ.depth(), \
                                                                                            new_circ.depth(), \
                                                                                            dist, \
                                                                                            fid_dist, \
                                                                                            self.total_time, \
                                                                                            len(self.qasm_blocks), \
                                                                                            self.cache, \
                                                                                            self.hit_rate, \
                                                                                            self.b_size, \
                                                                                            self.sc_threshold, \
                                                                                            self.project_name)
        
        
        if details == True:
            # output new unitaries
            #print("Output new unitaries")
            #new_unitary_blocks = self.get_new_unitary()
            #dirName = self.project_name + '/new_unitaries'
            #if not os.path.exists(dirName):
            #    os.mkdir(dirName)
            #for i in range(len(self.new_unitary_blocks)):
            #    filename = dirName + '/u_' + str(i)
            #    np.savetxt(filename, self.new_unitary_blocks[i])
                
            # compute distance for each block
            #print("Compute distance for each block")
            #for i in range(len(self.unitary_blocks)):
            #    if i % 10 == 0:
            #        print("process block - %s" % i)
            #    dist = sc.utils.matrix_distance(self.unitary_blocks[i], self.new_unitary_blocks[i])
            #    distance.append(dist)  
            #self.distance = distance
            
            # compute fidelity for each block
            print("Compute fidelity for each block")
            fidelity_blocks = []
            for i in range(len(self.new_qasm_blocks)):
                if i % 10 == 0:
                    print("process block - %s" % i)
                backend = Aer.get_backend('statevector_simulator')
                original_circuit = qk.QuantumCircuit.from_qasm_str(self.qasm_blocks[i])
                new_circuit = qk.QuantumCircuit.from_qasm_str(self.new_qasm_blocks[i])
                
                ori_job = execute(original_circuit, backend)
                block_state_ori = ori_job.result().get_statevector(original_circuit)
                new_job = execute(new_circuit, backend)
                block_state_new = new_job.result().get_statevector(new_circuit)
                
                fidelity = 1 - self.get_state_fidelity(original_circuit, new_circuit)
                
                fidelity_blocks.append(fidelity)
            self.fidelity_blocks = fidelity_blocks
            
            # blocks stats
            F = 1
            out_block_details += 'Block#\tQubits\t#ProgCNOTs\t#SwapCNOTs\t#SynCNOTs\t#1Q-Gates\tNew #1Q-Gates\tDepth\tNew Depth\tDistance\tInfidelity\tTime\n'
            for i in range(len(self.distance)):
                circ = QuantumCircuit.from_qasm_str(self.qasm_blocks[i])
                new_circ = QuantumCircuit.from_qasm_str(self.new_qasm_blocks[i])
                num_swap = self.qasm_blocks[i].count('swap')
                out_block_details += '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.2f\n' % (i, \
                                                                           circ.num_qubits, \
                                                                           circ.num_nonlocal_gates() - num_swap, \
                                                                           3 * num_swap, \
                                                                           new_circ.num_nonlocal_gates(), \
                                                                           circ.size() - circ.num_nonlocal_gates(), \
                                                                           new_circ.size() - new_circ.num_nonlocal_gates(), \
                                                                           circ.depth(), \
                                                                           new_circ.depth(), \
                                                                           self.distance[i], \
                                                                           fidelity_blocks[i], \
                                                                           self.exe_time[i]
                                                                          )
                
            
        
        # output stats.txt
        filename = self.project_name + '/block_stats.txt'
        out_file = open(filename, 'w')
        out_file.write(out_str_header)
        out_file.write(out_str)
        out_file.write("\n\n")
        out_file.write(str(self.valid_blocks))
        out_file.write("\n\n")
        out_file.write(self.str_time)
        out_file.write(out_block_details)
        out_file.close()
        
        filename = "results_all.txt"
        res_all = open(filename, 'a+')
        res_all.write(out_str)
        res_all.close()
        
    def run_partition_only(self):
        str_time = ''
        
        all_time = time.time()
        print("Synthesize with SC")
        sys.stdout.flush()
        
        if self.qubit_groups == None:
            start_time = time.time()
            print("Generate qubit groups")
            self.generate_qubit_groups()
            print("generate_qubit_groups:\t%.2f" % (time.time() - start_time))
            str_time += "generate_qubit_groups:\t%.2f\n" % (time.time() - start_time)
            sys.stdout.flush()
        else:
            print("Costumized qubit groups are given.")
        
        start_time = time.time()
        self.circ_partition()
        print("circ_partition:\t%.2f" % (time.time() - start_time))
        str_time += "circ_partition:\t%.2f\n" % (time.time() - start_time)
        sys.stdout.flush()
        
        list_cnot = []
        for i in range(len(self.qasm_blocks)):
            num_swap = self.qasm_blocks[i].count('swap')
            num_cnot = self.qasm_blocks[i].count('cx')
            total_cnot = 3 * num_swap + num_cnot
            list_cnot.append(total_cnot)
            
        
        out_str_header = 'Benchmark\tQubits\t#AvgCNOTs\t#HighCNOTs\t#LowCNOTs\t#Blocks\tBlock Size\n'
        out_str = '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (self.name, len(self.used_qubits), sum(list_cnot)/len(list_cnot), \
                                                max(list_cnot), min(list_cnot), len(self.qasm_blocks), self.b_size)
        
        filename = "partition_all.txt"
        res_all = open(filename, 'a+')
        #res_all.write(out_str_header)
        res_all.write(out_str)
        res_all.close()
        
        # output original qasm blocks
        dirName = self.name + '_original_qasm_blocks'
        if not os.path.exists(dirName):
            os.mkdir(dirName)
    
        for i in range(len(self.qasm_blocks)):
            filename = dirName + '/b_' + str(i) + '.qasm'
            out_file = open(filename, 'w')
            out_file.write(self.qasm_blocks[i])
            out_file.close()

    
        
