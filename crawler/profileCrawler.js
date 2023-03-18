'use strict';

const fs = require('fs');
const scrapedin = require('scrapedin')

let credential_file = fs.readFileSync('crawler/credential.json');
let credential = JSON.parse(credential_file);

const options = {
  email: credential.userName,
  password: credential.password
}

var args = process.argv.slice(2);
var profile_url = args[0]

console.log("Start crawling..." + profile_url)

scrapedin(options)
  .then((profileScraper) => profileScraper(profile_url))
  .then((profile) => {
    let data = JSON.stringify(profile, null, 2);

    fs.writeFile('data/profile.json', data, (err) => {
      if (err) {
        process.exit(1);
      }
      console.log("Crawling is done.")
      process.exit()
    });
  })


