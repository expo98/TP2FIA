import random
import copy
import numpy as np
import gymnasium as gym 
import os
from multiprocessing import Process, Queue

# CONFIG
ENABLE_WIND = False
WIND_POWER = 15.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = None
EPISODES = 1000
STEPS = 500

NUM_PROCESSES = os.cpu_count()
evaluationQueue = Queue()
evaluatedQueue = Queue()


nInputs = 8
nOutputs = 2
SHAPE = (nInputs,12,nOutputs)
GENOTYPE_SIZE = 0
for i in range(1, len(SHAPE)):
    GENOTYPE_SIZE += SHAPE[i-1]*SHAPE[i]

POPULATION_SIZE = 50
NUMBER_OF_GENERATIONS = 10
PROB_CROSSOVER = 0.7

  
PROB_MUTATION = 1.0/GENOTYPE_SIZE
STD_DEV = 0.1


ELITE_SIZE = 1


def network(shape, observation,ind):
    #Computes the output of the neural network given the observation and the genotype
    x = observation[:]
    for i in range(1,len(shape)):
        y = np.zeros(shape[i])
        for j in range(shape[i]):
            for k in range(len(x)):
                y[j] += x[k]*ind[k+j*len(x)]
        x = np.tanh(y)
    return x


def check_successful_landing(observation):
    #Checks the success of the landing based on the observation
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        return True
    return False

def objective_function(observation):
    #TODO: Implement your own objective function
    #Computes the quality of the individual based 
    #on the horizontal distance to the landing pad, the vertical velocity and the angle
    
    fitness = 0
    
    x = observation[0]
    y = observation[1]

    x_speed = observation[2]
    y_speed = observation[3]

    angle = observation[4]
    angle_speed = observation[5]

    left_leg_touching = observation[6]
    right_leg_touching = observation[7]
    
    # Em primeiro lugar definimos as penalizações
    


    #Aqui penalizamos a distancia 
    #distancia_zona =  np.sqrt( np.pow(x*10,2) + np.pow(y*10,2))
    #fitness -= pow(distancia_zona,2)

    # Aqui definimos as rewards

    if abs(x) <= 0.2 and left_leg_touching == 1 and right_leg_touching == 1:
        fitness += 300

    fitness += 200 - pow(abs(y_speed*20),2)
    fitness += 200 - pow(abs(angle*100),2)

    

    #if abs(y_speed) <= 0.2 and abs(y_speed) > 0.05:
        #fitness += 200
    
    #if abs(angle_speed) <= 0.2:
        #fitness += 100
    
    #if abs(angle) <= np.deg2rad(20):
        #fitness += 100


    return fitness, check_successful_landing(observation)

def simulate(genotype, render_mode = None, seed=None, env = None):
    #Simulates an episode of Lunar Lander, evaluating an individual
    env_was_none = env is None
    if env is None:
        env = gym.make("LunarLander-v3", render_mode =render_mode, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
        
    observation, info = env.reset(seed=seed)

    for _ in range(STEPS):
        prev_observation = observation
        #Chooses an action based on the individual's genotype
        action = network(SHAPE, observation, genotype)
        observation, reward, terminated, truncated, info = env.step(action)        

        if terminated == True or truncated == True:
            break
    
    if env_was_none:    
        env.close()

    return objective_function(prev_observation)

def evaluate(evaluationQueue, evaluatedQueue):
    #Evaluates individuals until it receives None
    #This function runs on multiple processes
    
    env = gym.make("LunarLander-v3", render_mode = None, 
        continuous=True, gravity=GRAVITY, 
        enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
        turbulence_power=TURBULENCE_POWER)    
    while True:
        ind = evaluationQueue.get()

        if ind is None:
            break
            
        ind['fitness'] = simulate(ind['genotype'], seed = None, env = env)[0]
                
        evaluatedQueue.put(ind)
    env.close()
    
def evaluate_population(population):
    #Evaluates a list of individuals using multiple processes
    for i in range(len(population)):
        evaluationQueue.put(population[i])
    new_pop = []
    for i in range(len(population)):
        ind = evaluatedQueue.get()
        new_pop.append(ind)
    return new_pop

def generate_initial_population():
    #Generates the initial population
    population = []
    for i in range(POPULATION_SIZE):
        #Each individual is a dictionary with a genotype and a fitness value
        #At this time, the fitness value is None
        #The genotype is a list of floats sampled from a uniform distribution between -1 and 1
        
        genotype = []
        for j in range(GENOTYPE_SIZE):
            genotype += [random.uniform(-1,1)]
        population.append({'genotype': genotype, 'fitness': None})
    return population

def parent_selection(population):
    #TODO
    #Select an individual from the population
    

    #Gets the total population fitness
    populationFitness = 0
    for individual in population:
        if individual['fitness'] > 0:
            populationFitness += individual['fitness']

    # If, by any chance, every element has negative fitness it will return a random element of the population
    if populationFitness <= 0:
        return copy.deepcopy(random.choice(population))
        

    #Selects a random probability
    randomProbability = random.random()

    for individual in population:
        #If an element has positive fitness it will get it's % over the whole 
        # positive Fitness population and subtract from the randomprobability
        if individual['fitness'] > 0:
            fitnessProbability = individual['fitness'] / populationFitness
            randomProbability -= fitnessProbability

        # If the random Probability now is 0 or less it means the element was selected
        if randomProbability <= 0:
            return copy.deepcopy(individual)
            
        
    return copy.deepcopy(random.choice(population))
    

def crossover(p1, p2):
    #TODO
    #Create an offspring from the individuals p1 and p2

    # Garante que pelo menos 1 elemento de cada pai é selecionado
    crossover_point = random.randint(1,GENOTYPE_SIZE-1)

    # Cria  o genotipo do offspring
    offspring_genotype = []

    #Define se p1 ou p2 tem o seu genotipo em primeiro lugar
    parent_order_selection = random.random()

    #Cria o novo genotiopo baseado no crossover_point aleatoriamente definido
    if parent_order_selection <= 0.5:
        offspring_genotype.extend(p1['genotype'][:crossover_point])
        offspring_genotype.extend(p2['genotype'][crossover_point:])

    else:
        offspring_genotype.extend(p2['genotype'][:crossover_point])
        offspring_genotype.extend(p1['genotype'][crossover_point:])


    #Cria o novo individuo baseado no genotype do offspring
    offspring = {'genotype': offspring_genotype, 'fitness': None}

    return offspring

def mutation(p):
    #TODO
    #Mutate the individual p

    mutated_individual = copy.deepcopy(p)
    individual_genotype = p['genotype']

    for i in range(len(individual_genotype)):
        mutation_probability = random.random()

        if mutation_probability < PROB_MUTATION:
            mutation_value = np.random.normal(0,STD_DEV)
            individual_genotype[i] += mutation_value

    return mutated_individual
    
def survival_selection(population, offspring):
    #reevaluation of the elite
    offspring.sort(key = lambda x: x['fitness'], reverse=True)
    p = evaluate_population(population[:ELITE_SIZE])
    new_population = p + offspring[ELITE_SIZE:]
    new_population.sort(key = lambda x: x['fitness'], reverse=True)
    return new_population    
        
def evolution():
    #Create evaluation processes
    evaluation_processes = []
    for i in range(NUM_PROCESSES):
        evaluation_processes.append(Process(target=evaluate, args=(evaluationQueue, evaluatedQueue)))
        evaluation_processes[-1].start()
    
    #Create initial population
    bests = []
    population = list(generate_initial_population())
    population = evaluate_population(population)
    population.sort(key = lambda x: x['fitness'], reverse=True)
    best = (population[0]['genotype']), population[0]['fitness']
    bests.append(best)
    
    #Iterate over generations
    for gen in range(NUMBER_OF_GENERATIONS):
        offspring = []
        
        #create offspring
        while len(offspring) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:
                p1 = parent_selection(population)
                p2 = parent_selection(population)
                ni = crossover(p1, p2)

            else:
                ni = parent_selection(population)
                
            ni = mutation(ni)
            offspring.append(ni)
            
        #Evaluate offspring
        offspring = evaluate_population(offspring)

        #Apply survival selection
        population = survival_selection(population, offspring)
        
        #Print and save the best of the current generation
        best = (population[0]['genotype']), population[0]['fitness']
        bests.append(best)
        print(f'Best of generation {gen}: {best[1]}')

    #Stop evaluation processes
    for i in range(NUM_PROCESSES):
        evaluationQueue.put(None)
    for p in evaluation_processes:
        p.join()
        
    #Return the list of bests
    return bests

def load_bests(fname):
    #Load bests from file
    bests = []
    with open(fname, 'r') as f:
        for line in f:
            fitness, shape, genotype = line.split('\t')
            bests.append(( eval(fitness),eval(shape), eval(genotype)))
    return bests

if __name__ == '__main__':
    
    evolve = True
    render_mode = 'human'
    if evolve:
        seeds = [964, 952, 364, 913, 140, 726, 112, 631, 881, 844, 965, 672, 335, 611, 457, 591, 551, 538, 673, 437, 513, 893, 709, 489, 788, 709, 751, 467, 596, 976]
        for i in range(30):    
            random.seed(seeds[i])
            bests = evolution()
            with open(f'log{i}.txt', 'w') as f:
                for b in bests:
                    f.write(f'{b[1]}\t{SHAPE}\t{b[0]}\n')

                
    else:
        #validate individual
        bests = load_bests('log0.txt')
        b = bests[-1]
        SHAPE = b[1]
        ind = b[2]
            
        ind = {'genotype': ind, 'fitness': None}
        
            
        ntests = 1000

        fit, success = 0, 0
        for i in range(1,ntests+1):
            f, s = simulate(ind['genotype'], render_mode=render_mode, seed = None)
            fit += f
            success += s
        print(fit/ntests, success/ntests)
