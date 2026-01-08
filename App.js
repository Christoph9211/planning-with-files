import React, { useState } from 'react';
import { View, ActivityIndicator } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import AuthContext from './src/AuthContext';
import Dashboard from './src/Dashboard';
import Quiz from './src/Quiz';
import Gamification from './src/Gamification';

const Stack = createStackNavigator();

export default function App() {
  const [isLoading, setIsLoading] = useState(true);
  const [userToken, setUserToken] = useState(null);

  // Simulate authentication check
  React.useEffect(() => {
    setTimeout(() => {
      setIsLoading(false);
      setUserToken('dummy-auth-token'); // Replace with actual auth logic
    }, 2000);
  }, []);

  if (isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  return (
    <AuthContext.Provider value={{ userToken, setUserToken }}>
      <NavigationContainer>
        <Stack.Navigator>
          {userToken ? (
            <>
              <Stack.Screen name="Dashboard" component={Dashboard} />
              <Stack.Screen name="Quiz" component={Quiz} />
              <Stack.Screen name="Gamification" component={Gamification} />
            </>
          ) : (
            <Stack.Screen name="Login" options={{ title: 'Welcome' }}>
              {props => <Dashboard {...props} />}
            </Stack.Screen>
          )}
        </Stack.Navigator>
      </NavigationContainer>
    </AuthContext.Provider>
  );
}
