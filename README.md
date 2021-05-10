# Self-Supervised-Music-Analysis
Self-Supervised Contrastive Learning of Music Spectrograms

## Dataset

Songs on the Billboard Year End Hot 100 were collected from the years 1960-2020. This list tracks the top songs of the US market for a given calendar year based on aggregating metrics including streaming plays, physical and digital purchases, radio plays, etc. In total the dataset includes 5737 songs, excluding some songs which could not be found and some which are duplicates across multiple years. It’s worth noting that the types of songs that are able to make it onto this sort of list represent a very narrow subset of the overall variety of the US music market, let alone the global music market. So while we can still learn some interesting things from this dataset, we shouldn’t mistake it for being representative of music in general.

Raw audio files were processed into spectrograms using a synchrosqueeze CWT algorithm from the Pyssqueeze python library. Some additional cleaning and postprocessing was done and the spectrograms were saved as grayscale images. These images are structured so that the Y axis which spans 256 pixels represents a range of frequencies from 30Hz – 12kHz with a log scale. The X axis represents time with a resolution of 200 pixels per second. Pixel intensity therefore encodes the signal energy at a particular frequency at a moment in time.


## Model and Training

A 30 layer ResNet styled CNN architecture was used as the primary feature extraction network. This was augmented with learned position embeddings along the frequency axis inserted at regular block intervals. Features were learned in a completely self-supervised fashion using Contrastive Learning. Matched pairs were taken as random 256x1024 pixel crops (corresponding to ~5 seconds of audio) from each song with no additional augmentations.

Output feature vectors have 512 channels representing a 64 pixel span (~0.3 seconds of audio).


## Results

The entirety of each song was processed via the feature extractor with the resulting song matrix averaged across the song length into a single vector. UMAP is used for visualization and HDBSCAN for cluster extraction producing the following plot:

![](../main/images/UMAPclusters.PNG)

Each color represents a cluster (numbered 0-16) of similar songs based on the learned features. Immediately we can see a very clear structure in the data, showing the meaningful features have been learned. We can also color the points by year of release:

![](../main/images/UMAPyears.PNG)

Points are colored form oldest (dark) to newest (light). As expected, the distribution of music has changed over the last 60 years. This gives us some confidence that the learned features are meaningful but let’s try a more specific test. A gradient boosting regressor model is trained on the learned features to predict the release year of a song.

![](../main/images/PredYear.PNG)

The model achieves an overall mean absolute error of ~6.2 years. The violin and box plots show the distribution of predictions for songs in each year. This result is surprisingly good considering we wouldn’t expect a model get anywhere near perfect. The plot shows some interesting trends in how the predicted median and overall variance shift from year to year. Notice, for example, the high variance and rapid median shift across the years 1990 to 2000 compared to the decades before and after. This hints at some potential significant changes in the structure of music during this decade. Those with a knowledge of modern musical history probably already have some ideas in mind. Again, it’s worth noting that this dataset represents generically popular music which we would expect to lag behind specific music trends (probably by as much as 5-10 years).

Let’s bring back the 17 clusters that were identified previously and look at the distribution of release years of songs in each cluster. The black grouping labeled -1 captures songs which were not strongly allocated to any particular cluster and is simply included for completeness.

![](../main/images/ClusterDistribution.PNG)

Here again we see some interesting trends of clusters emerging, peaking, and even dying out at various points in time. Aligning with out previous chart, we see four distinct clusters (7, 10, 11, 12) die off in the 90s while two brand new clusters (3, 4) emerge. Other clusters (8, 9, 15), interestingly, span most or all of the time range.

We can also look at the relative allocation of songs to clusters by year to get a better sense of the overall size of each cluster.

![](../main/images/StackedClusters.PNG)


## Cluster Samples

So what exactly are these clusters? I’ve provided links below to ten representative songs from each cluster so you can make your own qualitative evaluation. Before going further and listening to these songs I want to encourage you loosen your preconceived notions of musical genre. Popular conception of musical genres typically includes non-musical aspects like lyrics, theme, particular instruments, artist demographics, singer accent, year of release, marketing, etc. These aspects are not captured in the dataset and therefore not represented below but with an open ear you may find examples of songs that you considered to be different genres are actually quite musically similar.

**Cluster 0**
* [Manfred Mann's Earth Band - Blinded By The Light](https://open.spotify.com/track/7kNNylJ1kswWbHeRM6UDuE)
* [Elvis Presley - Bossa Nova Baby](https://open.spotify.com/track/22Z6ClJxSRovjPiswfCg3V)
* [Blondie - Rapture](https://open.spotify.com/track/6F2vo4sxRNQ58VYe3pdiaL)
* [The O'Jays - For the Love of Money](https://open.spotify.com/track/3p1JoOEhVkEnTaa4JzTMSk)
* [Stray Cats - Stray Cat Strut](https://open.spotify.com/track/5yogRsv5ggT6iCnFgvdpho)
* [Orleans - Still the One](https://open.spotify.com/track/2dtK02TSAuTvVYU2wGAVG0)
* [Babyface - When Can I See You](https://open.spotify.com/track/2zItQNJrVrTioXTXWiI2ed)
* [Color Me Badd - All 4 Love](https://open.spotify.com/track/4XmsMIMjvDIFEjeY3ycMzW)
* [Omarion - O](https://open.spotify.com/track/3gOBLrOvDovzHL4xBDQw0B)
* [Hootie & The Blowfish - Only Wanna Be with You](https://open.spotify.com/track/1OFKUn2VLafrHj7ybnap0Q)

**Cluster 1**
* [David Guetta - Turn Me On](https://open.spotify.com/track/6JOlNkT0QdHeZB0wPbI9IR)
* [Kelly Clarkson - Stronger](https://open.spotify.com/track/6D60klaHqbCl9ySc8VcRss)
* [Pitbull - International Love](https://open.spotify.com/track/62zFEHfAYl5kdHYOivj4BC)
* [Soulja Boy - Kiss Me Thru The Phone](https://open.spotify.com/track/2q4rjDy9WhaN3o9MvDbO21)
* [Soul II Soul - Keep On Movin'](https://open.spotify.com/track/7upgDi9C0pQn9HZzGfksJq)
* [Black Box - Strike It Up](https://open.spotify.com/track/742hY2twqAjwNYnKkQdilj)
* [Pitbull - Timber](https://open.spotify.com/track/3cHyrEgdyYRjgJKSOiOtcS)
* [Black Eyed Peas - Don't Lie](https://open.spotify.com/track/600qBKuhdgLqxZb1BqIE0T)
* [Lil Wayne - She Will](https://open.spotify.com/track/3FFcZZq3Z3EJrhUecwcMdG)
* [Paula Abdul - Rush Rush](https://open.spotify.com/track/015qd1I4v00JIoK7yOUgKC)

**Cluster 2**
* [Tara Kemp - Hold You Tight](https://open.spotify.com/track/0otWaD7P1jqYsb0qSHNo6J)
* [Allure - All Cried Out](https://open.spotify.com/track/1DViQw0p1vo0eAMRlUF4Lr)
* [Nelly Furtado - I'm Like A Bird](https://open.spotify.com/track/4sUoWHVnJl8z3t4zdqf6xB)
* [3OH!3 - My First Kiss](https://open.spotify.com/track/17tDv8WA8IhqE8qzuQn707)
* [WALK THE MOON - Shut Up and Dance](https://open.spotify.com/track/4kbj5MwxO1bq9wjT5g9HaA)
* [Third Eye Blind - Never Let You Go](https://open.spotify.com/track/1sxUaLi0G2vB7dl4ogtCxH)
* [P!nk - Stupid Girls](https://open.spotify.com/track/1Ab6RxeKl3e07zP7Get7CX)
* [Ellie Goulding - Burn](https://open.spotify.com/track/0xMd5bcWTbyXS7wPrBtZA6)
* [Dua Lipa - Break My Heart](https://open.spotify.com/track/017PF4Q3l4DBUiWoXk4OWT)
* [Charlie Puth - Attention](https://open.spotify.com/track/5cF0dROlMOK5uNZtivgu50)

**Cluster 3**
* [Migos - Stir Fry](https://open.spotify.com/track/2UVbBKQOdFAekPTRsnkzcf)
* [Ludacris - Get Back](https://open.spotify.com/track/3njpLvANriMsdv3dgADEad)
* [Waka Flocka Flame - No Hands](https://open.spotify.com/track/03tqyYWC9Um2ZqU0ZN849H)
* [Baby Bash - Cyclone](https://open.spotify.com/track/0x1LCpY9Rgeq97VPajm81B)
* [Yung Joc - I Know You See It](https://open.spotify.com/track/6giYNaycmjkbf7UmZ6RGtL)
* [DJ Snake - Taki Taki](https://open.spotify.com/track/4w8niZpiMy6qz1mntFA5uM)
* [Migos - Bad and Boujee](https://open.spotify.com/track/4Km5HrUvYTaSUfiSGPJeQR)
* [Miguel - Sky Walker](https://open.spotify.com/track/5WoaF1B5XIEnWfmb5NZikf)
* [Nelly - Hot In Herre](https://open.spotify.com/track/04KTF78FFg8sOHC1BADqbY)
* [Case - Touch Me Tease Me](https://open.spotify.com/track/4xrBjUq18fvXK68DJEi5XM)

**Cluster 4**
* [Desiigner - Panda](https://open.spotify.com/track/275a9yzwGB6ncAW4SxY7q3)
* [Chris Brown - Turn Up the Music](https://open.spotify.com/track/1RMRkCn07y2xtBip9DzwmC)
* [Stevie B - Dream About You](https://open.spotify.com/track/17u5YI0fhZGLwVwvrARGDE)
* [John Legend - Green Light](https://open.spotify.com/track/72by3Re4C3eVEBXvsUo0zV)
* [Eve - Let Me Blow Ya Mind](https://open.spotify.com/track/3RmKpob8xzv1pzHEQrMJah)
* [Mariah Carey - Touch My Body](https://open.spotify.com/track/2aEuXA1KswHlCGPOuPmCOW)
* [Jordin Sparks - One Step At a Time](https://open.spotify.com/track/5o4W6yWSJD9e9Ea8YC9WjF)
* [Surf Mesa - ily](https://open.spotify.com/track/62aP9fBQKYKxi7PDXwcUAS)
* [Drake - Jumpman](https://open.spotify.com/track/27GmP9AWRs744SzKcpJsTZ)
* [Khalid - Better](https://open.spotify.com/track/6zeeWid2sgw4lap2jV61PZ)

**Cluster 5**
* [Barbra Streisand - The Main EventFight](https://open.spotify.com/track/2g5jMKl1csCP8Ufp1rUnqC)
* [Peaches & Herb - Close Your Eyes](https://open.spotify.com/track/774ZoGBhlMSZd7nI2zojbc)
* [Air Supply - The One That You Love](https://open.spotify.com/track/3FDAd5vW4P7xe1GBNOLyfD)
* [Brian Hyland - Itsy Bitsy Teenie Weenie Yellow Polka Dot Bikini](https://open.spotify.com/track/3B3jI9LaQyOwrtjdlnNOw0)
* [Carrie Underwood - Inside Your Heaven](https://open.spotify.com/track/3w8xlZi49MQuQkyyB5bi5e)
* [LeAnn Rimes - How Do I Live](https://open.spotify.com/track/7BD50ATrF3Vab5FQy7vtK8)
* [Journey - Don't Stop Believin'](https://open.spotify.com/track/4bHsxqR3GMrXTxEPLuK5ue)
* [Rihanna - Cheers](https://open.spotify.com/track/46MDLc0Yip6xCMSsdePOAU)
* [Blues Traveler - Run-Around](https://open.spotify.com/track/500Tkm3vJmVtdUwdCVxCTb)
* [Phil Collins - Something Happened on the Way to Heaven](https://open.spotify.com/track/4ziqqoW1o3P5EhNqK6CPb1)

**Cluster 6**
* [NSYNC - Bye Bye Bye](https://open.spotify.com/track/62bOmKYxYg7dhrC6gH9vFn)
* [Eve 6 - Here's to the Night](https://open.spotify.com/track/4tgeQrPu5xkrnwErym2JsJ)
* [Nickelback - Gotta Be Somebody](https://open.spotify.com/track/06T10fEzN8ZCcqzQZYA184)
* [A Flock Of Seagulls - I Ran](https://open.spotify.com/track/2Rwux3fRY1xxl167muIo93)
* [Nickelback - If Everyone Cared](https://open.spotify.com/track/44w63XqGr3sATAzOnOySgF)
* [Stevie Nicks - Talk to Me](https://open.spotify.com/track/65ILbAZRAwZQ3omWKE0OIW)
* [The Temptations - The Way You Do The Things You Do](https://open.spotify.com/track/3496rr5XSGD6n1Z1OKXovb)
* [Billy Joel - It's Still Rock and Roll to Me](https://open.spotify.com/track/64UioB4Nmwgn2f4cbIpAkl)
* [Huey Lewis & The News - I Want A New Drug](https://open.spotify.com/track/0mZNKyrUmsrlDRoYHWsyMu)
* [Default - Wasting My Time](https://open.spotify.com/track/5dpAN1mjFPL38kh9kWsCiw)

**Cluster 7**
* [New Edition - If It Isn't Love](https://open.spotify.com/track/7JmPqImeW3kLoYVNBA9v11)
* [Leif Garrett - I Was Made for Dancin'](https://open.spotify.com/track/3kovBaZ1LGLH1PL31qG7cL)
* [James Taylor - You've Got a Friend](https://open.spotify.com/track/6zV8IpLvw0tkRSVCFQJB1y)
* [Three Dog Night - Mama Told Me](https://open.spotify.com/track/1CAO7hiNOxJRPW4nFv2aRO)
* [The Cars - You Might Think](https://open.spotify.com/track/35wVRTJlUu2kDkqXFegOKt)
* [Chicago - Feelin' Stronger Every Day](https://open.spotify.com/track/3hfqqSWS52Vucr4S0mLqUN)
* [Alice Cooper - I Never Cry](https://open.spotify.com/track/7y5mfSJLUmPYmrI5hcCWnT)
* [Ringo Starr - No-No Song](https://open.spotify.com/track/2YnZugg0pdEkAtSHR4dwFo)
* [James Taylor - Fire and Rain](https://open.spotify.com/track/1oht5GevPN9t1T3kG1m1GO)
* [Richard Marx - Endless Summer Nights](https://open.spotify.com/track/2iXH35MhsqO5Ry8a7iptpJ)

**Cluster 8**
* [J. Frank Wilson & The Cavaliers - Last Kiss](https://open.spotify.com/track/0gvcgQAaWojiOa5yP71Tw3)
* [Lou Christie - Two Faces Have I](https://open.spotify.com/track/1cpguRdD11njLNH2ZeWygE)
* [Herman's Hermits - Can't You Hear My Heartbeat](https://open.spotify.com/track/2tUP1fuo8EdOVz3Bw9r7yu)
* [Colbie Caillat - Fallin' For You](https://open.spotify.com/track/1le5KVGTF1xWf2aUj7ruLy)
* [Ciara - Promise](https://open.spotify.com/track/1pLdjo3lOBbMaoR4ZpybFH)
* [Van Halen - Can't Stop Lovin' You](https://open.spotify.com/track/6z3JD6IqVvu6TUBtCfQPbe)
* [Bobby Lewis - One Track Mind](https://open.spotify.com/track/7qsPhReEjuaWUGxgWZoL5l)
* [Luke Bryan - Country Girl](https://open.spotify.com/track/0yD66650JxhqKbW76C2qCo)
* [Three Dog Night - One](https://open.spotify.com/track/4ME2YQNThxEW63fxojzHvN)
* [Kool & The Gang - Cherish](https://open.spotify.com/track/7ktxXsACTciz3gAtuzd1uV)

**Cluster 9**
* [Edwin McCain - I'll Be](https://open.spotify.com/track/5K7AMlpc4796JRWXb26nCV)
* [Alan Jackson - It's Five O'Clock Somewhere](https://open.spotify.com/track/07KYRDFf8Q6sqj4PWCP9vh)
* [The Capris - There's A Moon Out Tonight](https://open.spotify.com/track/1BILDfIiSgFh6MvLHjq1Jo)
* [Manfred Mann - The Mighty Quinn](https://open.spotify.com/track/0cWPr2PvYTQERq88b4jGXw)
* [The Paris Sisters - I Love How You Love Me](https://open.spotify.com/track/3arpPGMRj2lEGYZPEjSWQv)
* [Tim McGraw - Live Like You Were Dying](https://open.spotify.com/track/7B1QliUMZv7gSTUGAfMRRD)
* [David Lee Roth - California Girls](https://open.spotify.com/track/4H3vuLX59XPqdtTpIesGyS)
* [Linda Ronstadt - Ooh Baby Baby](https://open.spotify.com/track/4FLEKG82bHeT2olTM8E2Fy)
* [Chicago - Colour My World](https://open.spotify.com/track/6F9z8Xe7EKyCSGexzi87ii)
* [Thompson Twins - King For A Day](https://open.spotify.com/track/0tBo8Uj7BmK3E5UBVhON2v)

**Cluster 10**
* [Roberta Flack - Making Love](https://open.spotify.com/track/1UpKK7U9ow2K1G6qNw9wnW)
* [Bobby McFerrin - Don't Worry Be Happy](https://open.spotify.com/track/4hObp5bmIJ3PP3cKA9K9GY)
* [Vanessa Williams - Colors Of The Wind](https://open.spotify.com/track/7xG1fakElLbxwyr9eyGEK6)
* [George Benson - On Broadway](https://open.spotify.com/track/2oFxITGDiTZ1JRwP8yinRt)
* [George Michael - Praying for Time](https://open.spotify.com/track/7CgRXXie9XIxVSkpKi40ID)
* [Alive 'N Kickin' - Tighter, Tighter](https://open.spotify.com/track/5OXATqrGVPG49BjEIY6yyM)
* [Cher - Bang Bang](https://open.spotify.com/track/6LLK7hZgXFYi5Jk4oRQvAl)
* [Hot Butter - Popcorn](https://open.spotify.com/track/6Z5wWOqBfk4G3bP1KF2Vbj)
* [Wilson Pickett - Funky Broadway](https://open.spotify.com/track/4Kj7BJGxHXqNiAGXVD2xAH)
* [Gary Wright - Love Is Alive](https://open.spotify.com/track/5vVuiXoHyRGxJeCaHUpgae)

**Cluster 11**
* [Cyndi Lauper - True Colors](https://open.spotify.com/track/2A6yzRGMgSQCUapR2ptm6A)
* [David Rose - The Stripper](https://open.spotify.com/track/00vH2PsEQTGRyJYhyIyDbr)
* [Starland Vocal Band - Afternoon Delight](https://open.spotify.com/track/3uLk0uQ4zMS26h89Of8XOD)
* [The Cufflinks - Tracy](https://open.spotify.com/track/09DaPXM8QrB3xJ7ulODVMY)
* [Celine Dion - Where Does My Heart Beat Now](https://open.spotify.com/track/0S7Vv5am0xbAXgJ2RFQR8S)
* [Tommy James & The Shondells - Mony Mony](https://open.spotify.com/track/23xk9Rf7oIHVUU1JvmXYFn)
* [Jimmy Soul - If You Wanna Be Happy](https://open.spotify.com/track/7D97JnBT73FWUh9KmRvP9M)
* [Cher - If I Could Turn Back Time](https://open.spotify.com/track/6mYrhCAGWzTdF8QnKuchXM)
* [The Manhattan Transfer - Boy From New York City](https://open.spotify.com/track/2KGTNgVuEJpcqPhxRCQhSy)
* [Little Peggy March - I Will Follow Him](https://open.spotify.com/track/3GQETOg4ZXyQ1jEFqfMoac)

**Cluster 12**
* [Don Henley - The Boys Of Summer](https://open.spotify.com/track/4gvea7UlDkAvsJBPZAd4oB)
* [Harold Melvin & The Blue Notes - Wake up Everybody](https://open.spotify.com/track/3J6XkNehF8ZoiGgnv5zjMo)
* [Culture Club - Do You Really Want To Hurt Me](https://open.spotify.com/track/1I6q6nwNjNgik1Qe8Oi0Y7)
* [Kris Kristofferson - Why Me](https://open.spotify.com/track/6uMKkOEkPJRumFvAzo5nr9)
* [Larry Verne - Mr. Custer](https://open.spotify.com/track/54UFvpOyoqlUu7N09UymMz)
* [Matt Monro - My Kind of Girl](https://open.spotify.com/track/4otoz66kYmUhJckTRaQVtu)
* [The Four Preps - More Money For You and Me](https://open.spotify.com/track/5X864LOuv4j0UZUWPqVVYV)
* [Lou Rawls - Your Good Thing](https://open.spotify.com/track/79VLN3Akfbtadc8IYuygQd)
* [Richard Harris - MacArthur Park](https://open.spotify.com/track/5DBEFajBEaHgbbwe7oN0KP)
* [Rocky Burnette - Tired of Toein' the Line](https://open.spotify.com/track/3E3K8IXR3o7XOBvaGE2e55)

**Cluster 13**
* [R.E.M. - Stand](https://open.spotify.com/track/22UhQSbYimuCnvI0Y07gFX)
* [The Irish Rovers - The Unicorn](https://open.spotify.com/track/6HvRHu2HtWgE720gG5v3wE)
* [Tom Jones - What's New Pussycat](https://open.spotify.com/track/4HjwGX3pJKJTeOSDpT6GCo)
* [Elton John - Can You Feel The Love Tonight](https://open.spotify.com/track/67HKtdqchK0rmODxsBeWT8)
* [Chesney Hawkes - The One and Only](https://open.spotify.com/track/69Tyiih00ZZKboHFnXp0VF)
* [Dion - Lovers Who Wander](https://open.spotify.com/track/3ApktgPHkNDBWaZCWvOKjK)
* [Ray Stevens - Misty](https://open.spotify.com/track/6oamuJZeT7F13iLwACdQ4X)
* [Lee Dorsey - Working in the Coal Mine](https://open.spotify.com/track/5GR1Jj5ahZtoR6WqyM5LP4)
* [Charles Wright & The Watts 103rd Street Rhythm Band - Express Yourself](https://open.spotify.com/track/6gQZKkphKIMxZgca5r7ImA)
* [Gary Puckett & The Union Gap - Young Girl](https://open.spotify.com/track/5SalzBUnKlQUP7VshLCHWW)

**Cluster 14**
* [U2 - Desire](https://open.spotify.com/track/4D01oA1mGouaAT7fubvKRT)
* [Prince - 1999](https://open.spotify.com/track/2QSUyofqpGDCo026OPiTBQ)
* [Tom Petty and the Heartbreakers - You Got Lucky](https://open.spotify.com/track/5odJF8z6Ref4rQ14fq0UIA)
* [New Edition - Cool It Now](https://open.spotify.com/track/5LkcAjqj5NOctNGi2qUjlw)
* [Blondie - Call Me](https://open.spotify.com/track/7HKxTNVlkHsfMLhigmhC0I)
* [Gnarls Barkley - Crazy](https://open.spotify.com/track/2N5zMZX7YeL1tico8oQxa9)
* [James Brown - Cold Sweat](https://open.spotify.com/track/3GWM2gYAWWBrrh1h9F8DEc)
* [Boyz II Men - Motownphilly](https://open.spotify.com/track/4LxIGAVfcQIw0zAQRyFhU8)
* [Michael Jackson - Billie Jean](https://open.spotify.com/track/5ChkMS8OtdzJeqyybCc9R5)
* [Digable Planets - Rebirth Of Slick](https://open.spotify.com/track/26q6YTrXt9l8qshIveiTX9)

**Cluster 15**
* [Jimmy Jones - Good Timin'](https://open.spotify.com/track/52BR7Vx8Zh3BENnHztAE5Q)
* [Emeli Sande - Next To Me](https://open.spotify.com/track/1Xsxp1SEOxuMzjrFZhtw8u)
* [Kyu Sakamoto - Sukiyaki](https://open.spotify.com/track/6meIeOX3DHdaCnaNw67abE)
* [Puddle Of Mudd - Blurry](https://open.spotify.com/track/6lSr3iZTC144PKhvbPFzMp)
* [Sean Paul - Get Busy](https://open.spotify.com/track/5qTvkDrSfvwDv6RBjjcfQr)
* [Black Eyed Peas - Rock That Body](https://open.spotify.com/track/5K5LbSTVuKKe1KGMNfBgIW)
* [Four Tops - Baby I Need Your Loving](https://open.spotify.com/track/3aCbwWCYCT3MJjZeUnlcp4)
* [Miranda Lambert - Bluebird](https://open.spotify.com/track/0kPeZAyIhIfeZNrtfjJGDB)
* [The Beach Boys - California Girls](https://open.spotify.com/track/6bJuuCtXYiwOcKT9s8uRh8)
* [Calvin Harris - Slide](https://open.spotify.com/track/6gpcs5eMhJwax4mIfKDYQk)

**Cluster 16**
* [Nick Cannon - Gigolo](https://open.spotify.com/track/1T1ZUKX4X87tVLaBGjwFv4)
* [John Mellencamp - Jack & Diane](https://open.spotify.com/track/43btz2xjMKpcmjkuRsvxyg)
* [Akon - Smack That](https://open.spotify.com/track/2kQuhkFX7uSVepCD3h29g5)
* [Sam Smith - Too Good At Goodbyes](https://open.spotify.com/track/3VlbOrM6nYPprVvzBZllE5)
* [Nelly - Shake Ya Tailfeather](https://open.spotify.com/track/4TJduXYW1Pg96EDNnfiwxJ)
* [N.E.R.D - Lemon](https://open.spotify.com/track/4PpuH4mxL0rD35mOWaLoKS)
* [Justin Timberlake - My Love](https://open.spotify.com/track/4NeOWqHmlrGRuBvsLJC9rL)
* [Ms. Lauryn Hill - Doo Wop](https://open.spotify.com/track/0uEp9E98JB5awlA084uaIg)
* [Robin Thicke - Blurred Lines](https://open.spotify.com/track/0n4bITAu0Y0nigrz3MFJMb)
* [Keith Sweat - I Want Her](https://open.spotify.com/track/24gxdUxufJ5eSamdYcPAKH)
