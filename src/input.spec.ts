import { expect, test } from "vitest";
import { COUNTRY_INPUT_FIELDS, describeCountryInput, describeFields, emailDomain } from "./input.js";

const profile = {
  id: 1,
  login: "ana",
  name: "Ana Silva",
  location: "Recife,\nBrasil",
  company: "  @acme  ",
  blog: "ana.dev",
  email: " Ana.Silva@GMX.DE ",
  twitter_username: "ana_s",
  bio: "Dev in\tSão Paulo",
};

test("lists the non-empty input fields in contract order, whitespace collapsed", () => {
  expect(COUNTRY_INPUT_FIELDS).toEqual(["location", "company", "blog", "email_domain", "twitter_username", "bio"]);
  expect(describeCountryInput(profile)).toBe(
    "location: Recife, Brasil\ncompany: @acme\nblog: ana.dev\nemail_domain: gmx.de\ntwitter_username: ana_s\nbio: Dev in São Paulo",
  );
});

test("never includes login, name or the local part of the email", () => {
  const text = describeCountryInput(profile);
  expect(text).not.toContain("ana\n");
  expect(text).not.toContain("Ana Silva");
  expect(text).not.toContain("Silva@");
});

test("omits blank and missing fields, and an email without a valid domain", () => {
  expect(describeCountryInput({ ...profile, location: " ", company: null, blog: undefined, email: "broken", twitter_username: "", bio: null })).toBe("");
  expect(describeCountryInput({ bio: "Berlin" })).toBe("bio: Berlin");
});

test("describeFields follows the given field order", () => {
  expect(describeFields(["bio", "email_domain"], { bio: "x", email: "a@b.org" })).toBe("bio: x\nemail_domain: b.org");
});

test("emailDomain keeps only the normalized part after the last @", () => {
  expect(emailDomain("Ana.Silva@GMX.DE")).toBe("gmx.de");
  expect(emailDomain("  a@b@Example.com. ")).toBe("example.com");
  expect(emailDomain("a@localhost")).toBeNull();
  expect(emailDomain("no-at-sign.de")).toBeNull();
  expect(emailDomain("a@")).toBeNull();
  expect(emailDomain("a@exa mple.com")).toBeNull();
  expect(emailDomain("a@.com")).toBeNull();
  expect(emailDomain("")).toBeNull();
  expect(emailDomain(null)).toBeNull();
  expect(emailDomain(undefined)).toBeNull();
});
