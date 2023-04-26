from neural import NeuralNet

xor_data = [
    ([0,0], [0]),
    ([0,1], [1]),
    ([1,0], [1]),
    ([1,1], [1]),
]

hidden = 8
print()
print(hidden)
xorn = NeuralNet(2,hidden,1)

xorn.train(xor_data)

print(xorn.test_with_expected(xor_data))

#print()
#print("\n\nTraining Voter Opinions")
#print()

#voter_opinion_data = [
#    ([ .9, .6, .8, .3, .1], [1]),
#    ([ .8, .8, .4, .6, .4], [1]),
#    ([ .7, .2, .4, .6, .3], [1]),
#    ([ .5, .5, .8, .4, .8], [0]),
#    ([ .3, .1, .6, .8, .8], [0]),
#    ([ .6, .3, .4, .3, .6], [0])
#]

#von = NeuralNet(5,6,1)
#von.train(voter_opinion_data)
#print(von.test_with_expected(voter_opinion_data))

#Evaluate with the test data
#test_data = [
#    ([ 1, 1, 1,.1,.1]),
#    ([.5,.2,.1,.7,.7]),
#    ([.8,.3,.3,.3,.8]),
#    ([.8,.3,.3,.8,.3]),
#    ([.9,.8,.8,.3,.6])
#]
#for t in range(len(test_data)):
#    print()
#    print(f"case 1: {test_data[t]} evalates to: {von.evaluate(test_data[t])}")
#    print()
