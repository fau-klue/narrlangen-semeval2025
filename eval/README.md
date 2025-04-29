# Confusion matrices for our best model
This folder contains the confusion matrices for the predictions of our best model versus the human annotations of the development set (the gold standard). The labels are the same for every language; asterisks in the images indicate labels not present in the development set of a particular language.

[English](EN_dev_confmat.pdf)
[Bulgarian](BG_dev_confmat.pdf)
[Portuguese](PT_dev_confmat.pdf)
[Russian](RU_dev_confmat.pdf)
[Hindi](HI_dev_confmat.pdf)

Since the actual labels are quite long, they have been abbreviated with numbers:

| **number** | **domain**          | **coarse-grained narrative**                      | **fine-grained narrative**                                             |
|------------|---------------------|---------------------------------------------------|------------------------------------------------------------------------|
| 1          | Climate change      | Amplifying Climate Fears                          | Amplifying existing fears of global warming                            |
| 2          | Climate change      | Amplifying Climate Fears                          | Doomsday scenarios for humans                                          |
| 3          | Climate change      | Amplifying Climate Fears                          | Earth will be uninhabitable soon                                       |
| 4          | Climate change      | Amplifying Climate Fears                          | Whatever we do it is already too late                                  |
| 5          | Climate change      | Amplifying Climate Fears                          | Other                                                                  |
| 6          | Climate change      | Climate change is beneficial                      | CO2 is beneficial                                                      |
| 7          | Climate change      | Climate change is beneficial                      | Temperature increase is beneficial                                     |
| 8          | Climate change      | Climate change is beneficial                      | Other                                                                  |
| 9          | Climate change      | Controversy about green technologies              | Nuclear energy is not climate friendly                                 |
| 10         | Climate change      | Controversy about green technologies              | Renewable energy is costly                                             |
| 11         | Climate change      | Controversy about green technologies              | Renewable energy is dangerous                                          |
| 12         | Climate change      | Controversy about green technologies              | Renewable energy is unreliable                                         |
| 13         | Climate change      | Controversy about green technologies              | Other                                                                  |
| 14         | Climate change      | Criticism of climate movement                     | Ad hominem attacks on key activists                                    |
| 15         | Climate change      | Criticism of climate movement                     | Climate movement is alarmist                                           |
| 16         | Climate change      | Criticism of climate movement                     | Climate movement is corrupt                                            |
| 17         | Climate change      | Criticism of climate movement                     | Other                                                                  |
| 18         | Climate change      | Criticism of climate policies                     | Climate policies are ineffective                                       |
| 19         | Climate change      | Criticism of climate policies                     | Climate policies are only for profit                                   |
| 20         | Climate change      | Criticism of climate policies                     | Climate policies have negative impact on the economy                   |
| 21         | Climate change      | Criticism of climate policies                     | Other                                                                  |
| 22         | Climate change      | Criticism of institutions and authorities         | Criticism of international entities                                    |
| 23         | Climate change      | Criticism of institutions and authorities         | Criticism of national governments                                      |
| 24         | Climate change      | Criticism of institutions and authorities         | Criticism of political organizations and figures                       |
| 25         | Climate change      | Criticism of institutions and authorities         | Criticism of the EU                                                    |
| 26         | Climate change      | Criticism of institutions and authorities         | Other                                                                  |
| 27         | Climate change      | Downplaying climate change                        | Climate cycles are natural                                             |
| 28         | Climate change      | Downplaying climate change                        | CO2 concentrations are too small to have an impact                     |
| 29         | Climate change      | Downplaying climate change                        | Human activities do not impact climate change                          |
| 30         | Climate change      | Downplaying climate change                        | Humans and nature will adapt to the changes                            |
| 31         | Climate change      | Downplaying climate change                        | Ice is not melting                                                     |
| 32         | Climate change      | Downplaying climate change                        | Sea levels are not rising                                              |
| 33         | Climate change      | Downplaying climate change                        | Temperature increase does not have significant impact                  |
| 34         | Climate change      | Downplaying climate change                        | Weather suggests the trend is global cooling                           |
| 35         | Climate change      | Downplaying climate change                        | Other                                                                  |
| 36         | Climate change      | Green policies are geopolitical instruments       | Climate-related international relations are abusive/exploitative       |
| 37         | Climate change      | Green policies are geopolitical instruments       | Green activities are a form of neo-colonialism                         |
| 38         | Climate change      | Green policies are geopolitical instruments       | Other                                                                  |
| 39         | Climate change      | Hidden plots by secret schemes of powerful groups | Blaming global elites                                                  |
| 40         | Climate change      | Hidden plots by secret schemes of powerful groups | Climate agenda has hidden motives                                      |
| 41         | Climate change      | Hidden plots by secret schemes of powerful groups | Other                                                                  |
| 42         | Climate change      | Questioning the measurements and science          | Data shows no temperature increase                                     |
| 43         | Climate change      | Questioning the measurements and science          | Greenhouse effect/carbon dioxide do not drive climate change           |
| 44         | Climate change      | Questioning the measurements and science          | Methodologies/metrics used are unreliable/faulty                       |
| 45         | Climate change      | Questioning the measurements and science          | Scientific community is unreliable                                     |
| 46         | Climate change      | Questioning the measurements and science          | Other                                                                  |
| 47         | Russo-Ukrainian War | Amplifying war-related fears                      | By continuing the war we risk WWIII                                    |
| 48         | Russo-Ukrainian War | Amplifying war-related fears                      | NATO should/will directly intervene                                    |
| 49         | Russo-Ukrainian War | Amplifying war-related fears                      | Russia will also attack other countries                                |
| 50         | Russo-Ukrainian War | Amplifying war-related fears                      | There is a real possibility that nuclear weapons will be employed      |
| 51         | Russo-Ukrainian War | Amplifying war-related fears                      | Other                                                                  |
| 52         | Russo-Ukrainian War | Blaming the war on others rather than the invader | The West are the aggressors                                            |
| 53         | Russo-Ukrainian War | Blaming the war on others rather than the invader | Ukraine is the aggressor                                               |
| 54         | Russo-Ukrainian War | Blaming the war on others rather than the invader | Other                                                                  |
| 55         | Russo-Ukrainian War | Discrediting the West, Diplomacy                  | Diplomacy does/will not work                                           |
| 56         | Russo-Ukrainian War | Discrediting the West, Diplomacy                  | The EU is divided                                                      |
| 57         | Russo-Ukrainian War | Discrediting the West, Diplomacy                  | The West does not care about Ukraine, only about its interests         |
| 58         | Russo-Ukrainian War | Discrediting the West, Diplomacy                  | The West is overreacting                                               |
| 59         | Russo-Ukrainian War | Discrediting the West, Diplomacy                  | The West is weak                                                       |
| 60         | Russo-Ukrainian War | Discrediting the West, Diplomacy                  | West is tired of Ukraine                                               |
| 61         | Russo-Ukrainian War | Discrediting the West, Diplomacy                  | Other                                                                  |
| 62         | Russo-Ukrainian War | Discrediting Ukraine                              | Discrediting Ukrainian government and officials and policies           |
| 63         | Russo-Ukrainian War | Discrediting Ukraine                              | Discrediting Ukrainian military                                        |
| 64         | Russo-Ukrainian War | Discrediting Ukraine                              | Discrediting Ukrainian nation and society                              |
| 65         | Russo-Ukrainian War | Discrediting Ukraine                              | Rewriting Ukraine’s history                                            |
| 66         | Russo-Ukrainian War | Discrediting Ukraine                              | Situation in Ukraine is hopeless                                       |
| 67         | Russo-Ukrainian War | Discrediting Ukraine                              | Ukraine is a hub for criminal activities                               |
| 68         | Russo-Ukrainian War | Discrediting Ukraine                              | Ukraine is a puppet of the West                                        |
| 69         | Russo-Ukrainian War | Discrediting Ukraine                              | Ukraine is associated with nazism                                      |
| 70         | Russo-Ukrainian War | Discrediting Ukraine                              | Other                                                                  |
| 71         | Russo-Ukrainian War | Distrust towards Media                            | Ukrainian media cannot be trusted                                      |
| 72         | Russo-Ukrainian War | Distrust towards Media                            | Western media is an instrument of propaganda                           |
| 73         | Russo-Ukrainian War | Distrust towards Media                            | Other                                                                  |
| 74         | Russo-Ukrainian War | Hidden plots by secret schemes of powerful groups | Other                                                                  |
| 75         | Russo-Ukrainian War | Negative Consequences for the West                | Sanctions imposed by Western countries will backfire                   |
| 76         | Russo-Ukrainian War | Negative Consequences for the West                | The conflict will increase the Ukrainian refugee flows to Europe       |
| 77         | Russo-Ukrainian War | Negative Consequences for the West                | Other                                                                  |
| 78         | Russo-Ukrainian War | Overpraising the West                             | NATO will destroy Russia                                               |
| 79         | Russo-Ukrainian War | Overpraising the West                             | The West belongs in the right side of history                          |
| 80         | Russo-Ukrainian War | Overpraising the West                             | The West has the strongest international support                       |
| 81         | Russo-Ukrainian War | Overpraising the West                             | Other                                                                  |
| 82         | Russo-Ukrainian War | Praise of Russia                                  | Praise of Russian military might                                       |
| 83         | Russo-Ukrainian War | Praise of Russia                                  | Praise of Russian President Vladimir Putin                             |
| 84         | Russo-Ukrainian War | Praise of Russia                                  | Russia has international support from a number of countries and people |
| 85         | Russo-Ukrainian War | Praise of Russia                                  | Russia is a guarantor of peace and prosperity                          |
| 86         | Russo-Ukrainian War | Praise of Russia                                  | Russian invasion has strong national support                           |
| 87         | Russo-Ukrainian War | Praise of Russia                                  | Other                                                                  |
| 88         | Russo-Ukrainian War | Russia is the Victim                              | Russia actions in Ukraine are only self-defence                        |
| 89         | Russo-Ukrainian War | Russia is the Victim                              | The West is russophobic                                                |
| 90         | Russo-Ukrainian War | Russia is the Victim                              | UA is anti-RU extremists                                               |
| 91         | Russo-Ukrainian War | Russia is the Victim                              | Other                                                                  |
| 92         | Russo-Ukrainian War | Speculating war outcomes                          | Russian army is collapsing                                             |
| 93         | Russo-Ukrainian War | Speculating war outcomes                          | Russian army will lose all the occupied territories                    |
| 94         | Russo-Ukrainian War | Speculating war outcomes                          | Ukrainian army is collapsing                                           |
| 95         | Russo-Ukrainian War | Speculating war outcomes                          | Other                                                                  |
| 96         | Other               |