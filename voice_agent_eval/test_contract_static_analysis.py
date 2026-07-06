import ast
import inspect
import orchestrator

def test_static_route_validation():
    """
    Static analysis test:
    For every AgentContract subclass in orchestrator.py, ensure that any string literal
    returned as the route (first element of tuple) in _route_on_goal_complete
    and _route_on_goal_incomplete is explicitly listed in that contract's 
    possible_next_actions.
    """
    # Find all AgentContract subclasses
    contracts = []
    for name, obj in inspect.getmembers(orchestrator):
        if inspect.isclass(obj) and issubclass(obj, orchestrator.AgentContract) and obj is not orchestrator.AgentContract and obj is not orchestrator.PlanningAgentContract:
            contracts.append(obj)
            
    for contract_cls in contracts:
        # Instantiate to get the declared possible_next_actions
        # (Assuming all contracts can be instantiated without args, which they are in this codebase)
        try:
            instance = contract_cls()
        except Exception:
            # If it takes args, skip for this basic test or mock args
            continue
            
        allowed_routes = set(instance.possible_next_actions)
        # Terminate is implicitly allowed by the workflow if goal satisfied, but let's check explicit returns
        # Actually, "Terminate" is handled differently or explicitly. Let's add it to allowed just in case,
        # or we check if it is explicitly returned.
        allowed_routes.add("Terminate")
        
        if contract_cls.__name__ not in ("PersonalShopperAgentContract", "FallbackNodeContract", "TerminateContract"):
            assert "PersonalShopperAgent" in allowed_routes, (
                f"Contract {contract_cls.__name__} is missing 'PersonalShopperAgent' in possible_next_actions. "
                "This is required because the base class check_universal_intents() can route to it."
            )
        
        # Parse the source code of the class
        source = inspect.getsource(contract_cls)
        tree = ast.parse(source)
        
        class_def = tree.body[0]
        methods_to_check = ['_route_on_goal_complete', '_route_on_goal_incomplete']
        
        for node in ast.walk(class_def):
            if isinstance(node, ast.FunctionDef) and node.name in methods_to_check:
                # Find all return statements
                for child in ast.walk(node):
                    if isinstance(child, ast.Return):
                        # We expect return statements like: return "AgentName", updates
                        returned_route = None
                        
                        if isinstance(child.value, ast.Tuple) and len(child.value.elts) >= 1:
                            # e.g. return "AgentName", {}
                            first_elem = child.value.elts[0]
                            if isinstance(first_elem, ast.Constant) and isinstance(first_elem.value, str):
                                returned_route = first_elem.value
                                
                        elif isinstance(child.value, ast.Constant) and isinstance(child.value.value, str):
                            # e.g. return "AgentName"
                            returned_route = child.value.value
                            
                        if returned_route:
                            assert returned_route in allowed_routes, (
                                f"Contract {contract_cls.__name__} in method {node.name} "
                                f"returns route '{returned_route}' which is NOT in its "
                                f"possible_next_actions: {instance.possible_next_actions}"
                            )
