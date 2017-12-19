using OpenAIGym
import Reinforce.action

env = GymEnv("Taxi-v2") # Taxi-v2 for 4x4

struct NewRandomPolicy <: AbstractPolicy end

function action(policy::NewRandomPolicy, r, s, A′)
	println("Current state: $s, Type: $(typeof(s))")
    rand(A′)
end

reset!(env)
ep = Episode(env, NewRandomPolicy())

i = 0
for (s, a, r, sp) in ep
    i+=1
    if i > 3 break end
end
