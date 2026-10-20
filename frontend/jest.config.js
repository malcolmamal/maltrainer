module.exports = {
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/main.ts',
    '!src/**/*.spec.ts',
    '!src/**/environments/**',
    '!src/**/*.d.ts'
  ],
  coverageThreshold: {
    global: { statements: 97, branches: 87, functions: 95, lines: 97 }
  },
  preset: 'jest-preset-angular',
  setupFilesAfterEnv: ['<rootDir>/setup-jest.ts'],
  testPathIgnorePatterns: ['<rootDir>/node_modules/', '<rootDir>/dist/'],
  moduleNameMapper: {
    '^src/(.*)$': '<rootDir>/src/$1',
  },
  transformIgnorePatterns: ['node_modules/(?!.*\\.mjs$|@stomp/rx-stomp|@stomp/stompjs)'],
};
