#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <memory>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <chrono>
#include <iomanip>

using namespace std;

// Abstract base class for all account types
class Account {
protected:
    string accountNumber;
    double balance;
public:
    Account(string accountNum, double initialBalance)
        : accountNumber(accountNum), balance(initialBalance) {}

    virtual void deposit(double amount) {
        if (amount <= 0) throw invalid_argument("Deposit amount must be positive.");
        balance += amount;
    }

    virtual void withdraw(double amount) {
        if (amount <= 0) throw invalid_argument("Withdrawal amount must be positive.");
        if (balance < amount) throw runtime_error("Insufficient funds.");
        balance -= amount;
    }

    virtual void displayAccountInfo() const {
        cout << "Account Number: " << accountNumber << "\nBalance: " << balance << endl;
    }

    virtual ~Account() = default;
    virtual void applyInterest() {}  // Empty by default for non-savings accounts
    virtual string getAccountType() const = 0;
};

// Savings Account with interest calculation
class SavingsAccount : public Account {
private:
    double interestRate;
public:
    SavingsAccount(string accountNum, double initialBalance, double rate)
        : Account(accountNum, initialBalance), interestRate(rate) {}

    void applyInterest() override {
        balance += balance * (interestRate / 100);
    }

    void displayAccountInfo() const override {
        cout << "Savings Account\n";
        Account::displayAccountInfo();
        cout << "Interest Rate: " << interestRate << "%" << endl;
    }

    string getAccountType() const override {
        return "Savings";
    }
};

// Checking Account with overdraft limit
class CheckingAccount : public Account {
private:
    double overdraftLimit;
public:
    CheckingAccount(string accountNum, double initialBalance, double overdraft)
        : Account(accountNum, initialBalance), overdraftLimit(overdraft) {}

    void withdraw(double amount) override {
        if (amount <= 0) throw invalid_argument("Withdrawal amount must be positive.");
        if (balance + overdraftLimit < amount) throw runtime_error("Insufficient funds, including overdraft.");
        balance -= amount;
    }

    void displayAccountInfo() const override {
        cout << "Checking Account\n";
        Account::displayAccountInfo();
        cout << "Overdraft Limit: " << overdraftLimit << endl;
    }

    string getAccountType() const override {
        return "Checking";
    }
};

// Loan Account that tracks loan balance and interest
class LoanAccount : public Account {
private:
    double interestRate;
    double loanAmount;
public:
    LoanAccount(string accountNum, double loanAmt, double rate)
        : Account(accountNum, 0), loanAmount(loanAmt), interestRate(rate) {}

    void applyInterest() override {
        loanAmount += loanAmount * (interestRate / 100);
    }

    void displayAccountInfo() const override {
        cout << "Loan Account\n";
        cout << "Loan Amount: " << loanAmount << "\nInterest Rate: " << interestRate << "%" << endl;
    }

    string getAccountType() const override {
        return "Loan";
    }
};

// Credit Card Account with limit and interest
class CreditCardAccount : public Account {
private:
    double creditLimit;
    double interestRate;
public:
    CreditCardAccount(string accountNum, double initialBalance, double limit, double rate)
        : Account(accountNum, initialBalance), creditLimit(limit), interestRate(rate) {}

    void applyInterest() override {
        if (balance > 0) {
            balance += balance * (interestRate / 100);
        }
    }

    void displayAccountInfo() const override {
        cout << "Credit Card Account\n";
        Account::displayAccountInfo();
        cout << "Credit Limit: " << creditLimit << "\nInterest Rate: " << interestRate << "%" << endl;
    }

    string getAccountType() const override {
        return "Credit Card";
    }
};

// Bank class that manages multiple accounts
class Bank {
private:
    vector<shared_ptr<Account>> accounts;
public:
    void addAccount(shared_ptr<Account> account) {
        accounts.push_back(account);
    }

    shared_ptr<Account> getAccount(const string& accountNumber) {
        for (auto& account : accounts) {
            if (account->accountNumber == accountNumber) {
                return account;
            }
        }
        throw runtime_error("Account not found.");
    }

    void displayAllAccounts() const {
        for (const auto& account : accounts) {
            account->displayAccountInfo();
        }
    }

    vector<shared_ptr<Account>> getAccounts() const {
        return accounts;
    }

    void applyInterestToAll() {
        for (auto& account : accounts) {
            account->applyInterest();
        }
    }
};

// File manager for reading and writing account data
class FileManager {
private:
    string filename;
public:
    FileManager(string fname) : filename(fname) {}

    void saveToFile(const Bank& bank) {
        ofstream outFile(filename, ios::out);
        if (!outFile) {
            throw runtime_error("Unable to open file for saving.");
        }

        for (const auto& account : bank.getAccounts()) {
            outFile << account->getAccountType() << ","
                    << account->accountNumber << ","
                    << account->balance << endl;
        }

        outFile.close();
    }

    void loadFromFile(Bank& bank) {
        ifstream inFile(filename, ios::in);
        if (!inFile) {
            throw runtime_error("Unable to open file for reading.");
        }

        string accountType, accountNumber;
        double balance;
        while (inFile >> accountType >> accountNumber >> balance) {
            if (accountType == "Savings") {
                bank.addAccount(make_shared<SavingsAccount>(accountNumber, balance, 2.5));
            } else if (accountType == "Checking") {
                bank.addAccount(make_shared<CheckingAccount>(accountNumber, balance, 500.0));
            } else if (accountType == "Loan") {
                bank.addAccount(make_shared<LoanAccount>(accountNumber, balance, 5.0));
            } else if (accountType == "Credit Card") {
                bank.addAccount(make_shared<CreditCardAccount>(accountNumber, balance, 1000.0, 18.0));
            }
        }

        inFile.close();
    }
};

// Logger class to log operations to a file
class Logger {
private:
    string filename;
public:
    Logger(string fname) : filename(fname) {}

    void logOperation(const string& operation) {
        ofstream outFile(filename, ios::app);
        if (!outFile) {
            throw runtime_error("Unable to open log file.");
        }

        auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
        outFile << put_time(localtime(&now), "%Y-%m-%d %H:%M:%S") << " - " << operation << endl;
        outFile.close();
    }
};

// User interface for interacting with the system
void showMenu() {
    cout << "\nBank System Menu:\n";
    cout << "1. Add Account\n";
    cout << "2. Display All Accounts\n";
    cout << "3. Apply Interest to All Accounts\n";
    cout << "4. Deposit Money\n";
    cout << "5. Withdraw Money\n";
    cout << "6. Save Data to File\n";
    cout << "7. Load Data from File\n";
    cout << "8. Exit\n";
}

void runBankSimulation() {
    Bank bank;
    FileManager fileManager("accounts.txt");
    Logger logger("operation_log.txt");

    // Add sample accounts
    bank.addAccount(make_shared<SavingsAccount>("12345", 1000.0, 2.5));
    bank.addAccount(make_shared<CheckingAccount>("67890", 2000.0, 500.0));
    bank.addAccount(make_shared<LoanAccount>("11111", 5000.0, 5.0));
    bank.addAccount(make_shared<CreditCardAccount>("22222", 1000.0, 1000.0, 18.0));

    logger.logOperation("Initial accounts created.");

    int choice;
    bool running = true;
    while (running) {
        showMenu();
        cout << "Choose an option: ";
        cin >> choice;

        switch (choice) {
        case 1: {
            string accountNumber, accountType;
            double balance;
            cout << "Enter account number: ";
            cin >> accountNumber;
            cout << "Enter account type (Savings, Checking, Loan, Credit Card): ";
            cin >> accountType;
            cout << "Enter initial balance: ";
            cin >> balance;

            if (accountType == "Savings") {
                bank.addAccount(make_shared<SavingsAccount>(accountNumber, balance, 2.5));
            } else if
