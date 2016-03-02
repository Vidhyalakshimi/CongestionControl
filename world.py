import random
import sys
import mdp
import environment
import util
import optparse
import subprocess
import math
from decimal import Decimal

class Networkworld(mdp.MarkovDecisionProcess):
  """
    Networkworld
  """
  def __init__(self):
    # layout
    self.livingReward = 0.0
    self.datarate = 10
    
    
        
  def setLivingReward(self, reward):
    """
    The (negative) reward for exiting "normal" states.
    
    Note that in the R+N text, this reward is on entering
    a state and therefore is not clearly part of the state's
    future rewards.
    """
    self.livingReward = reward
        
                                    
  def getPossibleActions(self, state):
    """
    Returns list of valid actions for 'state'.
    
    Note that you can request moves into walls and
    that "exit" states transition to the terminal
    state under the special action "done".
    """
    if state == ('low',8):
      return ()
    return (2.0,4.0,8.0,16.0,32.0)
    
  def getStates(self):
    """
    Return list of all states.
    """
    # The true terminal state.
    L_r = ["high", "medium", "low"]
    states = []
    for i in L_r:
        for j in range(0,8):
            states.append((i,j))
    states.append(('low',8))
    return states
        
  """
  why unnecessary action and nextState and reward is assigned based
  on state or nextState
  """

  def getReward(self, state, action, nextState):
    """
    Get reward for state, action, nextState transition.
    
    Note that the reward depends only on the state being
    departed (as in the R+N book examples, which more or
    less use this convention).
    """
    if state == ('low',7):
      return 1.0
    return self.livingReward
        
  def getStartState(self):
    
    return ('high',0)
    
  def isTerminal(self, state):
    """
    Only the TERMINAL_STATE state is *actually* a terminal state.
    The other "exit" states are technically non-terminals with
    a single action "exit" which leads to the true terminal state.
    This convention is to make the grids line up with the examples
    in the R+N textbook.
    """
    return state == ('low',8)
        
  
  def getTransitionState(self, state, action):
    """
    Returns list of (nextState, prob) pairs
    representing the states reachable
    from 'state' by taking 'action' along
    with their transition probabilities.          
    """        
  
    
    if state == ('low',7):
      successor = ('low',8)
      return successor
    """
    self.datarate = self.datarate/action
    prec = 1
    output = math.floor(self.datarate * (10 ** prec)) / (10 ** prec)
    """
    x = Decimal(self.datarate/action)
    self.datarate = round(x,2)
    passrate = str(self.datarate)+"Mbps"
    print "data rate" + str(self.datarate)
    cmd = './waf --run "scratch/simple-global-routing_tut --datarate='+passrate+'"'
    proc = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE )    
    for line in proc.stdout:
      if 'source' in line:
        sr1 = line.split(' ')
        sr = float(sr1[1])
        print "Source rate: "+str(sr1[1])
      elif 'Packet' in line:
        plr1 = line.split(' ')
        plr = float(plr1[1])
        print "Packet loss ratio: "+str(plr1[1])
      elif 'avg_' in line:
        avg_rec1 = line.split(' ')
        avg_rec = float(avg_rec1[1])
        print "avg_rec: "+str(avg_rec1[1])
      elif 'delay' in line:
        delay1 = line.split(' ')
        delay = float(delay1[1])
  print "Delay: "+ delay1[1]
    cmd1 = 'tcpdump -nn -tt -r simple-global-routing-0-1.pcap'
    proc1 = subprocess.Popen(cmd1,shell=True, stdout=subprocess.PIPE )
    for line in proc1.stdout:
        if '1.1' in line:
            break

    pdt_0_d = line.split(' ')
    pdt_0 = float(pdt_0_d[0])

    cmd2 = 'tcpdump -nn -tt -r simple-global-routing-3-1.pcap'
    proc2 = subprocess.Popen(cmd2,shell=True, stdout=subprocess.PIPE )
    for line in proc2.stdout:
        if '1.1' in line:
            pdt_1_d = line.split(' ')
            pdt_1 = float(pdt_1_d[0])
            if pdt_1 - pdt_0 > 0:
                break

    pdt = pdt_1 - pdt_0

    rtt = 2*pdt + delay
    print "pdt: "+ str(pdt) 
    print "rtt: "+ str(rtt)

    #stateformulation()
    if plr < 0.25: #congestion level - cl
      cl = 'low'
    elif plr > 0.25 and plr < 0.75:
      cl = 'medium'
    else: 
      cl = 'high'
    #avg_rec
    if avg_rec < 0.75*sr:
      bits = '0'
    else:
      bits = '1'
    #pdt
    if pdt > 0.005758:
      bits = bits+'0'   
    else:
      bits = bits+'1'
    #rtt
    if rtt > 2.032524:
      bits = bits+'0'
    else:
      bits = bits+'1'
  
    successor = (cl,int(bits,2))
    return successor
    '''
    if state == ('low',7):
        successor = ('low',8)
        return successor
        
    self.datarate = self.datarate/action
    prec = 1
    output = math.floor(self.datarate * (10 ** prec)) / (10 ** prec)
    passrate = str(output)+"Mbps"
    print "data rate" + str(self.datarate)
    cmd = './waf --run "scratch/simple-global-routing_tut --datarate='+passrate+'"'
    proc = subprocess.Popen(cmd,shell=True, stdout=subprocess.PIPE )
    for line in proc.stdout:
        if 'source' in line:
            sr1 = line.split(' ')
            sr = float(sr1[1])
            print "Packet rate: "+str(sr1[1])
        elif 'Packet' in line:
            plr1 = line.split(' ')
            plr = float(plr1[1])
            print "Packet loss ratio: "+str(plr1[1])
        elif 'avg_' in line:
            avg_rec1 = line.split(' ')
            avg_rec = float(avg_rec1[1])
            print "avg_rec: "+str(avg_rec1[1])
        elif 'pdt' in line:
            pdt1 = line.split(' ')
            pdt = float(pdt1[1])
            print "pdt: "+str(pdt1[1])
        elif 'rtt' in line:
            rtt1 = line.split(' ')
            rtt = float(rtt1[1])
            print "rtt: "+str(rtt1[1])
    #stateformulation()
    if plr < 0.25: #congestion level - cl
        cl = 'low'
    elif plr > 0.25 and plr < 0.75:
        cl = 'medium'
    else:
        cl = 'high'
    #avg_rec
                                                                                                                                        
    if avg_rec < 0.75*sr:
        bits = '0'
    else:
        bits = '1'
                                                                                                                                                        
    #pdt
    if pdt > 9.00233:
        bits = bits+'0'
    else:
        bits = bits+'1'
                                                                                                                                                                        
    #rtt
    if rtt > 18.0251:
        bits = bits+'0'
    else:
        bits = bits+'1'
                                                                                                                                                                                        
    successor = (cl,int(bits,2))
    return successor
  '''
  '''
  def __aggregate(self, statesAndProbs):
    counter = util.Counter()
    for state, prob in statesAndProbs:
      counter[state] += prob
    newStatesAndProbs = []
    for state, prob in counter.items():
      newStatesAndProbs.append((state, prob))
    return newStatesAndProbs
        
  def __isAllowed(self, y, x):
    if y < 0 or y >= self.grid.height: return False
    if x < 0 or x >= self.grid.width: return False
    return self.grid[x][y] != '#'

'''
class NetworkworldEnvironment(environment.Environment):
    
  def __init__(self,networkWorld):
    self.networkWorld = networkWorld
    self.reset()
            
  def getCurrentState(self):
    return self.state
        
  def getPossibleActions(self, state):        
    return self.networkWorld.getPossibleActions(state)
        
  def doAction(self, action):
    successor = self.networkWorld.getTransitionState(self.state, action) 
    state = self.getCurrentState()
    
    reward = self.networkWorld.getReward(state, action, successor)
    self.state = successor
    return (successor, reward)
    raise 'doAction failure'

  def displayQValues(self, agent, currentState = None, message = None):
    if message != None: print message
    qValues = util.Counter()
    states = self.networkWorld.getStates()
    print "Q Values: \n"
    for state in states:
      for action in self.networkWorld.getPossibleActions(state):
        qValues[(state, action)] = agent.getQValue(state, action)
        if qValues[(state, action)] != 0:
            print "State: "+ str(state) + "Action: " + str(action)
            print qValues[(state, action)]


  def reset(self):
    self.state = self.networkWorld.getStartState()
    self.networkWorld.datarate = 10
'''
def makeGrid(gridString):
  width, height = len(gridString[0]), len(gridString)
  grid = Grid(width, height)
  for ybar, line in enumerate(gridString):
    y = height - ybar - 1
    for x, el in enumerate(line):
      grid[x][y] = el
  return grid    
'''      
def getMyNetwork():
  return Networkworld()

def getUserAction(state, actionFunction):
  """
  Get an action from the user (rather than the agent).
  
  Used for debugging and lecture demos.
  """
  import graphicsUtils
  action = None
  while True:
    keys = graphicsUtils.wait_for_keys()
    if 'Up' in keys: action = 'north'
    if 'Down' in keys: action = 'south'
    if 'Left' in keys: action = 'west'
    if 'Right' in keys: action = 'east'
    if 'q' in keys: sys.exit(0)
    if action == None: continue
    break
  actions = actionFunction(state)
  if action not in actions:
    action = actions[0]
  return action

def printString(x): print x

def runEpisode(agent, environment, discount, decision, message, pause, episode):
  returns = 0
  totalDiscount = 1.0
  environment.reset()
  if 'startEpisode' in dir(agent): agent.startEpisode()
  message("BEGINNING EPISODE: "+str(episode)+"\n")
  while True:

    # DISPLAY CURRENT STATE
    state = environment.getCurrentState()
    print "Current state" 
    print state
    #display(state)
    pause() 
    
    # END IF IN A TERMINAL STATE
    actions = environment.getPossibleActions(state)
    if len(actions) == 0:
      message("EPISODE "+str(episode)+" COMPLETE: RETURN WAS "+str(returns)+"\n")
      return returns
    
    # GET ACTION (USUALLY FROM AGENT)
    action = decision(state)
    if action == None:
      raise 'Error: Agent returned None action'
    
    # EXECUTE ACTION
    nextState, reward = environment.doAction(action)
    message("Started in state: "+str(state)+
            "\nTook action: "+str(action)+
            "\nEnded in state: "+str(nextState)+
            "\nGot reward: "+str(reward)+"\n")    
    # UPDATE LEARNER
    if 'observeTransition' in dir(agent): 
        agent.observeTransition(state, action, nextState, reward)
    
    returns += reward * totalDiscount
    totalDiscount *= discount

  if 'stopEpisode' in dir(agent):
    agent.stopEpisode()

def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-d', '--discount',action='store',
                         type='float',dest='discount',default=0.9,
                         help='Discount on future (default %default)')
    optParser.add_option('-r', '--livingReward',action='store',
                         type='float',dest='livingReward',default=0.0,
                         metavar="R", help='Reward for living for a time step (default %default)')
    optParser.add_option('-n', '--noise',action='store',
                         type='float',dest='noise',default=0.2,
                         metavar="P", help='How often action results in ' +
                         'unintended direction (default %default)' )
    optParser.add_option('-e', '--epsilon',action='store',
                         type='float',dest='epsilon',default=0.3,
                         metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    optParser.add_option('-l', '--learningRate',action='store',
                         type='float',dest='learningRate',default=0.5,
                         metavar="P", help='TD learning rate (default %default)' )
    optParser.add_option('-i', '--iterations',action='store',
                         type='int',dest='iters',default=10,
                         metavar="K", help='Number of rounds of value iteration (default %default)')
    optParser.add_option('-k', '--episodes',action='store',
                         type='int',dest='episodes',default=1,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-g', '--grid',action='store',
                         metavar="G", type='string',dest='grid',default="BookGrid",
                         help='Grid to use (case sensitive; options are BookGrid, BridgeGrid, CliffGrid, MazeGrid, default %default)' )
    optParser.add_option('-w', '--windowSize', metavar="X", type='int',dest='gridSize',default=150,
                         help='Request a window width of X pixels *per grid cell* (default %default)')
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="random",
                         help='Agent type (options are \'random\', \'value\' and \'q\', default %default)')
    optParser.add_option('-t', '--text',action='store_true',
                         dest='textDisplay',default=False,
                         help='Use text-only ASCII display')
    optParser.add_option('-p', '--pause',action='store_true',
                         dest='pause',default=False,
                         help='Pause GUI after each time step when running the MDP')
    optParser.add_option('-q', '--quiet',action='store_true',
                         dest='quiet',default=False,
                         help='Skip display of any learning episodes')
    optParser.add_option('-s', '--speed',action='store', metavar="S", type=float,
                         dest='speed',default=1.0,
                         help='Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)')
    optParser.add_option('-m', '--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent')
    optParser.add_option('-v', '--valueSteps',action='store_true' ,default=False,
                         help='Display each step of value iteration')

    opts, args = optParser.parse_args()
    
    if opts.manual and opts.agent != 'q':
      print ('## Disabling Agents in Manual Mode (-m) ##')
      opts.agent = None

    # MANAGE CONFLICTS
    if opts.textDisplay or opts.quiet:
    # if opts.quiet:      
      opts.pause = False
      # opts.manual = False
      
    if opts.manual:
      opts.pause = True
      
    return opts


  
if __name__ == '__main__':
  
  opts = parseOptions()

  ###########################
  # GET THE GRIDWORLD
  ###########################

  import world
  mdpFunction = getattr(world, "getMyNetwork")
  mdp = mdpFunction()
  mdp.setLivingReward(opts.livingReward)
  env = world.NetworkworldEnvironment(mdp)

  
  ###########################
  # GET THE DISPLAY ADAPTER
  ###########################

  

  ###########################
  # GET THE AGENT
  ###########################

  import valueIterationAgents, qlearningAgents
  a = None
  if opts.agent == 'value':
    a = valueIterationAgents.ValueIterationAgent(mdp, opts.discount, opts.iters)
  elif opts.agent == 'q':
    #env.getPossibleActions, opts.discount, opts.learningRate, opts.epsilon
    #simulationFn = lambda agent, state: simulation.GridworldSimulation(agent,state,mdp)
    networkWorldEnv = NetworkworldEnvironment(mdp)
    actionFn = lambda state: mdp.getPossibleActions(state)
    qLearnOpts = {'gamma': opts.discount, 
                  'alpha': opts.learningRate, 
                  'epsilon': opts.epsilon,
                  'actionFn': actionFn}
    a = qlearningAgents.QLearningAgent(**qLearnOpts)
  elif opts.agent == 'random':
    # # No reason to use the random agent without episodes
    if opts.episodes == 0:
      opts.episodes = 10
    class RandomAgent:
      def getAction(self, state):
        return random.choice(mdp.getPossibleActions(state))
      def getValue(self, state):
        return 0.0
      def getQValue(self, state, action):
        return 0.0
      def getPolicy(self, state):
        "NOTE: 'random' is a special policy value; don't use it in your code."
        return 'random'
      def update(self, state, action, nextState, reward):
        pass      
    a = RandomAgent()
  else:
    if not opts.manual: raise 'Unknown agent type: '+opts.agent
    
    
  ###########################
  # RUN EPISODES
  ###########################
  # DISPLAY Q/V VALUES BEFORE SIMULATION OF EPISODES
  if not opts.manual and opts.agent == 'value':
    if opts.valueSteps:
      for i in range(opts.iters):
        tempAgent = valueIterationAgents.ValueIterationAgent(mdp, opts.discount, i)
        display.displayValues(tempAgent, message = "VALUES AFTER "+str(i)+" ITERATIONS")
        display.pause()        
    
    display.displayValues(a, message = "VALUES AFTER "+str(opts.iters)+" ITERATIONS")
    display.pause()
    display.displayQValues(a, message = "Q-VALUES AFTER "+str(opts.iters)+" ITERATIONS")
    display.pause()
    
  

  # FIGURE OUT WHAT TO DISPLAY EACH TIME STEP (IF ANYTHING)
  displayCallback = lambda x: None
  if not opts.quiet:
    if opts.manual and opts.agent == None: 
      displayCallback = lambda state: display.displayNullValues(state)
    else:
      if opts.agent == 'random': displayCallback = lambda state: display.displayValues(a, state, "CURRENT VALUES")
      if opts.agent == 'value': displayCallback = lambda state: display.displayValues(a, state, "CURRENT VALUES")
      if opts.agent == 'q': displayCallback = lambda state: display.displayQValues(a, state, "CURRENT Q-VALUES")

  messageCallback = lambda x: printString(x)
  if opts.quiet:
    messageCallback = lambda x: None

  # FIGURE OUT WHETHER TO WAIT FOR A KEY PRESS AFTER EACH TIME STEP
  pauseCallback = lambda : None
  if opts.pause:
    pauseCallback = lambda : display.pause()

  # FIGURE OUT WHETHER THE USER WANTS MANUAL CONTROL (FOR DEBUGGING AND DEMOS)  
  if opts.manual:
    decisionCallback = lambda state : getUserAction(state, mdp.getPossibleActions)
  else:
    decisionCallback = a.getAction  
    
  # RUN EPISODES
  if opts.episodes > 0:
    print
    print ("RUNNING", opts.episodes, "EPISODES")
    print
  returns = 0
  for episode in range(1, opts.episodes+1): 
    returns += runEpisode(a, env, opts.discount, decisionCallback, messageCallback, pauseCallback, episode)
  if opts.episodes > 0:
    print
    print ("AVERAGE RETURNS FROM START STATE: "+str((returns+0.0) / opts.episodes))
    print
    print
    
  # DISPLAY POST-LEARNING VALUES / Q-VALUES
  if opts.agent == 'q' and not opts.manual:
    env.displayQValues(a, message = "Q-VALUES AFTER "+str(opts.episodes)+" EPISODES")
    #display.pause()
    #display.displayValues(a, message = "VALUES AFTER "+str(opts.episodes)+" EPISODES")
    #display.pause()
    
   