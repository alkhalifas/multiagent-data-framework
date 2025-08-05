from multiagent import create_default_agents

manager, agents = create_default_agents()

print(manager.generate_trends())

answer = manager.ask("What were the total sales in March 2025?")
print(answer)
