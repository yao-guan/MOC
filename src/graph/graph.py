import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio
import textwrap
from src.graph.node import Node
from src.agents.agent_registry import AgentRegistry
from src.prompt.prompt_set_registry import PromptSetRegistry
from src.llm.profile_embedding import get_sentence_embedding
from src.gnn.gcn import GCN,MLP
from torch_geometric.utils import dense_to_sparse
from sklearn.metrics.pairwise import cosine_similarity
from src.llm.price import cost_count_conpressed
class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                use_neighbor_summary: bool = False,
                neighbor_hops: int = 1,
                ism_r: float = 1.0,
                ism_epsilon: float = 0.01,
                ism_kppa: int = 45,
                use_cot: bool = True,  
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.use_cot = use_cot
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()
        self.gcn = GCN(self.features.size(1)*2,16,self.features.size(1))
        self.mlp = MLP(384,16,16)

        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) 
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  

        # Neighbor summary settings
        self.use_neighbor_summary = use_neighbor_summary
        self.neighbor_hops = neighbor_hops
        self.ism_r = ism_r
        self.ism_epsilon = ism_epsilon
        self.ism_kppa = ism_kppa


    def construct_adj_matrix(self):
        role_connect:List[Tuple[str,str]] = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes,num_nodes))
        role_2_id = {}
        
        for edge in role_connect:
            in_role, out_role = edge
            role_2_id[in_role] = []
            role_2_id[out_role] = []
        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            role_2_id[role].append(i)
            
        for edge in role_connect:
            in_role,out_role = edge
            in_ids = role_2_id[in_role]
            out_ids = role_2_id[out_role]
            for in_id in in_ids:
                for out_id in out_ids:
                    role_adj[in_id][out_id] = 1
        
        edge_index, edge_weight = dense_to_sparse(role_adj)
        return edge_index
    
    #node Encoder
    def construct_features(self):
        features = []
        for node_id in self.nodes:
            role = self.nodes[node_id].role
            profile = self.prompt_set.get_description(role)
            feature = get_sentence_embedding(profile)
            features.append(feature)
        features = torch.tensor(np.array(features))
        return features
    
    def construct_new_features(self, query):
        query_embedding = torch.tensor(get_sentence_embedding(query))
        query_embedding = query_embedding.unsqueeze(0).repeat((self.num_nodes,1))
        new_features = torch.cat((self.features,query_embedding),dim=1)
        return new_features
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                kwargs["use_cot"] = self.use_cot
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,):
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:

        log_probs = 0
        new_features = self.construct_new_features(input['task'])
        logits = self.gcn(new_features,self.role_adj_matrix)
        logits = self.mlp(logits)
        self.spatial_logits = logits @ logits.t()
        self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input, graph=self),timeout=max_time) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks
    

    def get_neighbors_by_hops(self, node_id: str, max_hops: int = 1) -> Dict[int, List[Node]]:
        """
        Get neighbors organized by hop distance (including 1-hop neighbors).
        Maintains topological order: k-hop neighbors are organized according to their (k-1)-hop parents.
        
        Args:
            node_id: The ID of the target node
            max_hops: Maximum number of hops to collect
            
        Returns:
            Dictionary mapping hop distance to list of neighbor nodes
            {1: [nodes at 1 hop], 2: [nodes at 2 hops], 3: [nodes at 3 hops], ...}
            Note: 1-hop neighbors are included, topological order preserved
        """
        if node_id not in self.nodes or max_hops < 1:
            return {}
        
        target_node = self.nodes[node_id]
        neighbors_by_hop = {}
        
        current_level = [target_node]  
        visited = {target_node}
        
        for hop in range(1, max_hops + 1):
            next_level = []  
            for node in current_level:
                for predecessor in node.spatial_predecessors:
                    if predecessor not in visited:
                        next_level.append(predecessor)  
                        visited.add(predecessor)
            
            if not next_level:
                break  
            
            if hop >= 1:
                neighbors_by_hop[hop] = next_level  
            
            current_level = next_level
        
        return neighbors_by_hop

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embedding vector.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = get_sentence_embedding(text)
        return np.array(embedding)

    async def merge_multiple_messages(self, messages: List[str], kppa: int) -> str:
        """
        Merge multiple messages via LLM summarization with compression.

        Tries multiple compression strategies and selects the output with the highest
        similarity to the original messages.

        Args:
            messages: List of 6 elements [id_i, role_i, content_i, id_j, role_j, content_j].
            kppa: Target compression percentage for the summary length (default 45).

        Returns:
            Merged and compressed summary string.
        """
        if not messages:
            return ""
        if len(messages) == 3:
            return messages[-1]
        
        # k_percent = 45
        k_percent = kppa
        
        import ollama
        client = ollama.AsyncClient()
        
        id_i, role_i, content_i, id_j, role_j, content_j = messages
        contents = [content_i, content_j]
        messages_set = f"""AGENT_1: [ID: {id_i} | Role: {role_i}]
{content_i}

AGENT_2: [ID: {id_j} | Role: {role_j}]
{content_j}"""

        prompts = [
            # 1. (Focus: Narrative Synthesis)
            f"""Synthesize the messages from AGENT_1 and AGENT_2 into a single, cohesive update. 
Target length: approximately {k_percent}% of the original token count. 
Task: Merge overlapping information and deduplicate common findings while preserving both agents' distinct contributions. 
Constraint: Do not add new information. Output ONLY the synthesized text with no preamble.

{messages_set}""",

            # 2. (Focus: Logical Integrity)
            f"""Merge the communications from AGENT_1 and AGENT_2. 
Target length: roughly {k_percent}% of the original volume. 
Task: Compress the text but strictly retain the complete reasoning chain and all logical dependencies leading to the conclusion. 
Constraint: Do not add new information. Output ONLY the synthesized text with no preamble.

{messages_set}""",

            # 3. (Focus: Technical Precision)
            f"""Consolidate the data from AGENT_1 and AGENT_2 into a high-density summary. 
Target length: around {k_percent}% of original tokens. 
Task: Ensure zero-loss for all Agent IDs, technical parameters, formulas, and specific values. Strip away all conversational fillers. 
Constraint: Do not add new information. Output ONLY the synthesized text with no preamble.

{messages_set}""",

            # 4. (Focus: Actionable Intelligence)
            f"""Combine AGENT_1 and AGENT_2 messages into a "telegram-style" actionable update. 
Target length: approximately {k_percent}% volume. 
Task: Prioritize actionable data and final decisions. Use shorthand where possible while maintaining source attribution for key facts. 
Constraint: Do not add new information. Output ONLY the synthesized text with no preamble.

{messages_set}""",

            # 5. (Focus: Deduplication & Structure)
            f"""Integrate the content from AGENT_1 and AGENT_2 while maintaining any structural headers. 
Target length: about {k_percent}% of the original count. 
Task: Identify and merge redundant statements between the two agents to maximize information density per token. 
Constraint: Do not add new information. Output ONLY the synthesized text with no preamble.

{messages_set}"""
        ]
        
        try:
            outputs = []
            for i, prompt in enumerate(prompts):
                print(f"[Summary] Trying strategy {i+1}/5...")
                response = await client.chat(
                    model="gemma2:9b",
                    messages=[{'role': 'user', 'content': prompt}],
                    stream=False,
                    options={
                        'temperature': 0.1,
                    },
                )
                merged = response['message']['content']
                prompt_tokens = response.get('prompt_eval_count', 0)
                completion_tokens = response.get('eval_count', 0)
                cost_count_conpressed(prompt_tokens,completion_tokens)
                print(f"conpressed_prompt_tokens:{prompt_tokens},conpressed_completion_tokens:{completion_tokens}")
                outputs.append(merged.strip())
            
            original_embeddings = []
            for msg in contents:
                embedding = self.encode_text(msg)
                original_embeddings.append(embedding)
            
            best_output = None
            best_similarity = -1
            
            for i, output in enumerate(outputs):
                output_embedding = np.array(get_sentence_embedding(output))
                
                total_similarity = 0
                for orig_emb in original_embeddings:
                    similarity = cosine_similarity(
                        output_embedding.reshape(1, -1), 
                        orig_emb.reshape(1, -1)
                    )[0][0]
                    total_similarity += similarity
                
                print(f"[Summary] Strategy {i+1} similarity: {total_similarity:.4f}")
                
                if total_similarity > best_similarity:
                    best_similarity = total_similarity
                    best_output = output
            
            print(f"[Summary] Selected best strategy with similarity: {best_similarity:.4f}, avg: {best_similarity/2:.4f}")
            return best_output
            
        except Exception as e:
            print(f"[Summary] LLM merge failed: {e}, falling back to concatenation")
            return "\n\n".join(messages)
        
    async def iterative_semantic_merging_with_clustering(
        self, 
        neighbors: List[tuple],  # (hop, id, role, message)
        r: float = 1.0,
        epsilon: float = 0.01,
        kppa: int = 45
    ) -> str:
        """
        ISM Algorithm - Semantic merging based on message count and similarity threshold.

        Core ideas:
        1. Compute max message count K = |C|/k + r*k, where |C| is the original message count.
        2. Each iteration merges disjoint message pairs whose similarity >= max_similarity - epsilon.
        3. Maintain index order after merging (sort by original_idx and reassign consecutive indices).
        4. Roles are merged by concatenation; Messages are merged via LLM summarization.

        Args:
            neighbors: List of tuples (hop, id, role, message) from neighbor agents.
            r: Controls the target message count K (default 1.0).
            epsilon: Similarity threshold margin subtracted from max similarity (default 0.01).
            kppa: LLM summarization compression percentage (default 45).

        Returns:
            Aggregated context string.
        """
        if not neighbors:
            return ""
        
        max_items = int(len(neighbors) / self.neighbor_hops + r * self.neighbor_hops)
        print(f"[ISM] Computed K: {max_items} (r={r})")
        
        if len(neighbors) <= max_items:
            print(f"[ISM] Message count ({len(neighbors)}) within K ({max_items}), no merge needed")
            aggregated_parts = []
            for hop, id, role, message in neighbors:
                aggregated_parts.append(f"Agent {id},role is {role},output is:\n{message}")
            return "\n\n".join(aggregated_parts)

        P = []  # List of tuples: (h_vector, hop, id, role, content, original_idx)
        
        print(f"[ISM Phase 1] Encoding {len(neighbors)} neighbor nodes...")
        
        for i, (hop, id, role, message) in enumerate(neighbors):
            role_embedding = self.encode_text(role)
            content_embedding = self.encode_text(message)
            P.append((content_embedding, hop, id, role, message, i))
        
        print(f"[ISM Phase 1] Encoding done, {len(P)} items total")
        iteration = 0
        while len(P) > max_items:
            iteration += 1
            print(f"[ISM Phase 2] Iteration {iteration}, current: {len(P)} items")
            
            vectors = np.array([item[0] for item in P])
            sim_matrix = cosine_similarity(vectors)
            
            np.fill_diagonal(sim_matrix, -1)
            
            max_similarity = np.max(sim_matrix)
            similarity_threshold = max_similarity - epsilon

            print(f"[ISM Phase 2] Max similarity: {max_similarity:.4f}, threshold: {similarity_threshold:.4f}")
            merge_candidates = []
            for i in range(len(sim_matrix)):
                for j in range(i + 1, len(sim_matrix)):
                    if sim_matrix[i, j] >= similarity_threshold:
                        merge_candidates.append((i, j, sim_matrix[i, j]))
            
            merge_candidates.sort(key=lambda x: x[2], reverse=True)
            
            print(f"[ISM Phase 2] Found {len(merge_candidates)} candidate pairs (similarity >= {similarity_threshold:.4f})")
            
            if not merge_candidates:
                print(f"[ISM Phase 2] No pairs meet threshold, stopping iteration")
                break
            
            selected_pairs = []
            used_indices = set()
            
            for i_idx, j_idx, similarity in merge_candidates:
                if i_idx not in used_indices and j_idx not in used_indices:
                    selected_pairs.append((i_idx, j_idx, similarity))
                    used_indices.add(i_idx)
                    used_indices.add(j_idx)
            
            print(f"[ISM Phase 2] Selected {len(selected_pairs)} disjoint pairs to merge")
            if not selected_pairs:
                print(f"[ISM Phase 2] No disjoint pairs found, stopping iteration")
                break
            
            new_P = []
            merged_indices = set()
            
            for i_idx, j_idx, similarity in selected_pairs:
                h_vec_i, hop_i, id_i, role_i, content_i, orig_i = P[i_idx]
                h_vec_j, hop_j, id_j, role_j, content_j, orig_j = P[j_idx]
                merged_hop = min(hop_i, hop_j)
                merged_orig_idx = max(orig_i, orig_j)
                
                merged_role = f"{role_i} & {role_j}"
                merged_id = f"{id_i} & {id_j}"
                merged_content = await self.merge_multiple_messages([id_i, role_i, content_i, id_j, role_j, content_j], kppa)   # gy0121
                print(f"[ISM Phase 2] Merged pair ({i_idx}, {j_idx}), hop={merged_hop}, idx={merged_orig_idx}, sim={similarity:.4f}")
                
                role_embedding_i = self.encode_text(role_i)
                role_embedding_j = self.encode_text(role_j)
                role_embedding = role_embedding_i + role_embedding_j
                role_embedding = role_embedding / (np.linalg.norm(role_embedding) + 1e-8)
                content_embedding = self.encode_text(merged_content)
                
                new_P.append((content_embedding, merged_hop, merged_id, merged_role, merged_content, merged_orig_idx))
                
                merged_indices.add(i_idx)
                merged_indices.add(j_idx)
            
            for idx, item in enumerate(P):
                if idx not in merged_indices:
                    new_P.append(item)
            
            new_P.sort(key=lambda x: x[5])
            
            P = []
            for i, (h_vector, hop, id, role, content, _) in enumerate(new_P):
                P.append((h_vector, hop, id, role, content, i))
            
            print(f"[ISM Phase 2] Iteration {iteration} done, {len(P)} items remaining")
        
        aggregated_parts = []
        for _, hop, id, role, content, _ in P:
            aggregated_parts.append(f"Agent {id}, role is {role},output is:\n{content}")
        
        aggregated_context = "\n\n".join(aggregated_parts)
        
        return aggregated_context

    async def get_neighbor_summary_with_ism(self, node_id: str, query: str) -> str:
        """
        Get neighbor summary using Iterative Semantic Merging (ISM).
        This is an enhanced version of get_neighbor_summary that uses ISM algorithm.
        
        Args:
            node_id: The ID of the target node
            query: The current task/query
            
        Returns:
            ISM-aggregated summary of neighbor outputs
        """
        if not self.use_neighbor_summary:
            return ""
        
        print(f"[DEBUG] get_neighbor_summary_with_ism called for node {node_id}")
        neighbors_by_hop = self.get_neighbors_by_hops(node_id, self.neighbor_hops)
        
        if not neighbors_by_hop:
            print(f"[DEBUG] No neighbors found, returning empty summary")
            return ""
        
        print(f"[DEBUG] Found neighbors at hops: {list(neighbors_by_hop.keys())}")
        
        neighbor_data = []
        
        for hop in sorted(neighbors_by_hop.keys(), reverse=True):
            neighbors = neighbors_by_hop[hop]
            
            if not neighbors:
                continue
            
            for neighbor in neighbors:
                output = neighbor.outputs[-1] if neighbor.outputs else ""
                output_str = str(output)
                neighbor_data.append((hop, neighbor.id,neighbor.role,output_str))
        
        if not neighbor_data:
            print(f"[DEBUG] No neighbor data collected, returning empty summary")
            return ""
        
        print(f"[DEBUG] Collected {len(neighbor_data)} neighbor messages")
        
        aggregated_context = await self.iterative_semantic_merging_with_clustering(
            neighbors=neighbor_data,
            r=self.ism_r,
            epsilon=self.ism_epsilon,
            kppa=self.ism_kppa
        )
        
        if aggregated_context:
            result = f"{aggregated_context}"
        else:
            result = ""
        
        return result

def min_max_norm(tensor:torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
    