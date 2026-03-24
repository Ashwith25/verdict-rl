import metaworld
mt1 = metaworld.MT1("reach-v3")
env = mt1.train_classes["reach-v3"]()
env.set_task(mt1.train_tasks[0])
o = env.reset()
a = env.action_space.sample()
out = env.step(a)
print(len(out), type(out))