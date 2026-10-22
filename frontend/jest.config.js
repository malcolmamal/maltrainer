module.exports = {
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/main.ts',
    '!src/**/*.spec.ts',
    '!src/**/environments/**',
    '!src/**/*.d.ts'
  ],
  coverageThreshold: {
    global: { statements: 98, branches: 92, functions: 96, lines: 98 }
  },
  preset: 'jest-preset-angular',
  setupFilesAfterEnv: ['<rootDir>/setup-jest.ts'],
  testPathIgnorePatterns: ['<rootDir>/node_modules/', '<rootDir>/dist/'],
  moduleNameMapper: {
    '^src/(.*)$': '<rootDir>/src/$1',
  },
  transformIgnorePatterns: ['node_modules/(?!.*\\.mjs$|@stomp/rx-stomp|@stomp/stompjs)'],
};
