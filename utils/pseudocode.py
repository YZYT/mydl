def Save(new_individual):
    # select a victim to eliminate, 
    # making room for the new individual
    victim = Select_victim()
    # eliminate one and save one,
    # keep total number unchanged
    del victim
    save new_individual


# select an individual to perform SGD.
candidate = Select_candidate()
# use one batch to calculate 
# the loss of that candidate
l = loss_fn(candidate)

# update the candidate via gradient w.r.t $l,
# to generate a new individual$.
new_individual = GD(candidate, l)
Save(new_individual)

if nums % T == 0:
    # choose some individuals as fathers
    fathers = Select_fathers()

    # generate another new individual
    # by evolution of the new_individual.
    evo_individual = evolve(new_individual, fathers)
    Save(evo_individual)


