


both = {} # distance in both directions


xr = len(image)
        #print xr
yr = len(image[0])

    
centerx = xr/2
centery = yr/2

        #print yr

for i in range(0, xr):
    for j in range(0, yr): 
                #print i
                #print j
        
        #initialize dictionary for distance to center
        pixel = image[i, j]
        lbl = segments[i, j]
        both[lbl] = 0


        both[lbl] += sqrt( abs(i - (xr/2))**2 + abs(j - (yr/2))**2 )

        